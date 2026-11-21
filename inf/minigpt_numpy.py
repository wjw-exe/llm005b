# -*- coding: utf-8 -*-
"""MiniGPT（hayder86al/tinystories-50m-instruct）的 numpy 手写推理实现。

结构逐项对齐仓库 modeling_minigpt.py（已逐行核对）：
  * token_emb: Embedding(8000, 768)，pos_emb: Embedding(1024, 768) 可学习绝对位置
  * 6 个 TransformerBlock，pre-LN：
        x = x + attn(ln1(x));  x = x + ff(ln2(x))
  * ln1/ln2/ln_f: nn.LayerNorm(768)，默认 eps=1e-5、带 affine（weight+bias）
  * attn: qkv = Linear(768, 2304, bias=False)，proj = Linear(768, 768, bias=False)
          SDPA(is_causal=True) 默认缩放 1/sqrt(head_dim)=1/8
  * ff  : fc = Linear(768, 3072)（有 bias）-> GELU()（exact erf 版，非 tanh 近似）
          -> proj = Linear(3072, 768)（有 bias）
  * head: Linear(768, 8000, bias=False)，与 token_emb 权重绑定（weight tying）
  * 前向取最后 ctx=1024 个 token，位置从 0 重新编号

推理侧新增 KV Cache 增量解码：只在 prefill 时算全序列，之后每步只处理 1 个 token，
历史 K/V 存于 cache；当长度将超过 1024 时按窗口滑出最旧 token 并重建 cache，
与参考实现"每步 crop 到 ctx 并从 0 编号"在数学上等价。
"""
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from . import ops as _ops
    from .safetensors_lite import load_safetensors, read_header
    from .unigram_tok import UnigramTokenizer
except ImportError:  # 作为脚本直接运行时
    import ops as _ops  # type: ignore
    from safetensors_lite import load_safetensors, read_header  # type: ignore
    from unigram_tok import UnigramTokenizer  # type: ignore

DT = np.float32
NEG = DT(-1e30)


# --------------------------- exact GELU（erf 版） ---------------------------
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_P = 0.3275911


def _erf(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    t = 1.0 / (1.0 + _P * ax)
    poly = ((((_A[4] * t + _A[3]) * t + _A[2]) * t + _A[1]) * t + _A[0]) * t
    y = 1.0 - poly * np.exp(-ax * ax)
    return np.sign(x) * y.astype(DT)


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """nn.GELU() 默认 approximation='none'：0.5x(1+erf(x/sqrt2))。"""
    return (DT(0.5) * x * (DT(1.0) + _erf(x * DT(0.7071067811865476)))).astype(DT)


def _ln(x, g, b):
    y, _ = _ops.layernorm_forward(x, g, b, eps=1e-5)
    return y


def _lin(x, W, b=None):
    y, _ = _ops.linear_forward(x, W, b)
    return y


class KVCache:
    """预分配的 KV Cache：每层 k/v 各 (1, H, ctx, Dh)，按绝对位置写入，零拼接。"""

    __slots__ = ("k", "v", "len", "ctx")

    def __init__(self, n_layer: int, n_head: int, ctx: int, dh: int):
        self.k = [np.zeros((1, n_head, ctx, dh), dtype=DT) for _ in range(n_layer)]
        self.v = [np.zeros((1, n_head, ctx, dh), dtype=DT) for _ in range(n_layer)]
        self.len = 0
        self.ctx = ctx

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k) + sum(a.nbytes for a in self.v)


class MiniGPTInstruct:
    """权重 + 分词器 + 手写前向（含 KV Cache）。"""

    def __init__(self, model_dir: str, verbose: bool = True):
        self.dir = model_dir
        cfg_p = os.path.join(model_dir, "config.json")
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.n_layer = int(cfg["n_layer"])
        self.n_head = int(cfg["n_head"])
        self.n_embd = int(cfg["n_embd"])
        self.ctx = int(cfg["context_len"])
        self.vocab = int(cfg["vocab_size"])
        self.dh = self.n_embd // self.n_head
        self.eos_id = int(cfg.get("eos_token_id", 1))
        self.pad_id = int(cfg.get("pad_token_id", 0))
        self.arch = cfg.get("architectures")

        self.tok = UnigramTokenizer(os.path.join(model_dir, "tokenizer.json"))
        self.p = self._load_weights(verbose)
        self.param_count = sum(int(v.size) for v in self.p.values())
        self.uniq_count = self.param_count  # 见 _load_weights 内的去重统计
        self.peak_mem = 0.0
        # 预热一次 1-token 前向：把 BLAS 线程池与 212MB 权重页提前拉起，
        # 避免首个真实问题的 prefill 混入冷启动开销（实测冷启动可达 2s+）
        self.forward([self.pad_id], None, 0)

    # --------------------------- 权重加载与校验 ---------------------------
    def _load_weights(self, verbose: bool) -> Dict[str, np.ndarray]:
        path = os.path.join(self.dir, "model.safetensors")
        header, _, _ = read_header(path)
        raw = load_safetensors(path)

        strip = lambda k: k.split(".", 1)[1] if k.startswith("minigpt.") else k
        # 一次性拷出 memmap 视图并转成 C 连续 float32：
        # 避免首次前向因缺页中断从磁盘逐页读 212MB（实测会把首 token 拖到 2s+）
        got = {strip(k): np.ascontiguousarray(v, dtype=DT) for k, v in raw.items()}
        del raw

        P: Dict[str, np.ndarray] = {}
        P["token_emb"] = got["token_emb.weight"]
        P["pos_emb"] = got["pos_emb.weight"]
        P["ln_f.weight"] = got["ln_f.weight"]
        P["ln_f.bias"] = got["ln_f.bias"]
        # 注意：safetensors 中 nn.Linear 权重是 PyTorch 布局 (out, in)，
        # 手写算子 linear_forward 需要 (in, out)，因此这里统一转置。
        for i in range(self.n_layer):
            s = f"blocks.{i}."
            P[s + "ln1.weight"] = got[s + "ln1.weight"]
            P[s + "ln1.bias"] = got[s + "ln1.bias"]
            P[s + "qkv"] = np.ascontiguousarray(got[s + "attn.qkv.weight"].T)
            P[s + "proj"] = np.ascontiguousarray(got[s + "attn.proj.weight"].T)
            P[s + "ln2.weight"] = got[s + "ln2.weight"]
            P[s + "ln2.bias"] = got[s + "ln2.bias"]
            P[s + "fc"] = np.ascontiguousarray(got[s + "ff.fc.weight"].T)
            P[s + "fc.b"] = got[s + "ff.fc.bias"]
            P[s + "ffp"] = np.ascontiguousarray(got[s + "ff.proj.weight"].T)
            P[s + "ffp.b"] = got[s + "ff.proj.bias"]

        # 形状校验
        C, V, T = self.n_embd, self.vocab, self.ctx
        expect = {"token_emb": (V, C), "pos_emb": (T, C),
                  "ln_f.weight": (C,), "ln_f.bias": (C,)}
        for i in range(self.n_layer):
            s = f"blocks.{i}."
            expect.update({s + "ln1.weight": (C,), s + "ln1.bias": (C,),
                           s + "qkv": (C, 3 * C), s + "proj": (C, C),
                           s + "ln2.weight": (C,), s + "ln2.bias": (C,),
                           s + "fc": (C, 4 * C), s + "fc.b": (4 * C,),
                           s + "ffp": (4 * C, C), s + "ffp.b": (C,)})
        bad = [f"{k}: got {P[k].shape} want {expect[k]}"
               for k in expect if tuple(P[k].shape) != tuple(expect[k])]
        if bad:
            raise ValueError("形状校验失败: " + "; ".join(bad))

        self.missing_keys: List[str] = []
        self.unexpected_keys: List[str] = []
        known = {f"blocks.{i}.{n}" for i in range(self.n_layer)
                 for n in ("ln1.weight", "ln1.bias", "attn.qkv.weight",
                           "attn.proj.weight", "ln2.weight", "ln2.bias",
                           "ff.fc.weight", "ff.fc.bias", "ff.proj.weight",
                           "ff.proj.bias")}
        known |= {"token_emb.weight", "pos_emb.weight", "ln_f.weight", "ln_f.bias"}
        for k in known:
            if k not in got:
                self.missing_keys.append(k)
        for k in got:
            if k not in known:
                self.unexpected_keys.append(k)

        # 权重绑定校验：head.weight 应与 token_emb 相同
        self.tied_diff = None
        if "head.weight" in got:
            d = float(np.abs(got["head.weight"] - got["token_emb.weight"]).max())
            self.tied_diff = d
            self.unexpected_keys = [k for k in self.unexpected_keys if k != "head.weight"]
        self.raw_array_count = len(header)

        if verbose:
            print(f"[权重] {path}")
            print(f"[权重] safetensors 数组 {self.raw_array_count} 个，"
                  f"实际匹配进图 {len(P)} 个张量")
            print(f"[权重] 缺失键 {len(self.missing_keys)} 个 {self.missing_keys}")
            print(f"[权重] 多余键 {len(self.unexpected_keys)} 个 {self.unexpected_keys}")
            print(f"[权重] 唯一参数(去重后实际进图) "
                  f"{sum(int(v.size) for v in P.values()):,}，"
                  f"文件内合计(含 tied 重复) {sum(int(v.size) for v in got.values()):,}")
            if self.tied_diff is not None:
                print(f"[权重] head.weight 与 token_emb 最大差异 {self.tied_diff:.3e}"
                      f"（{'完全绑定' if self.tied_diff == 0 else '绑定但有数值差'}）")
        # 统一转 float32 连续
        for k, v in P.items():
            P[k] = np.ascontiguousarray(v, dtype=DT)
        return P

    # ------------------------------ 前向 ------------------------------
    def _attn(self, x, i, cache: KVCache, pos0: int):
        """x:(1,T,C)；把本步 k/v 写入 cache 的 pos0..pos0+T 区间，再对 [0,pos0+T) 做因果注意力。"""
        P = self.p
        s = f"blocks.{i}."
        B, T, C = x.shape
        qkv = _lin(x, P[s + "qkv"])
        q = np.ascontiguousarray(qkv[..., :C])
        k = np.ascontiguousarray(qkv[..., C:2 * C])
        v = np.ascontiguousarray(qkv[..., 2 * C:])
        H, Dh = self.n_head, self.dh
        sh = lambda t: t.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        kk, vv = cache.k[i], cache.v[i]
        kk[:, :, pos0:pos0 + T, :] = k
        vv[:, :, pos0:pos0 + T, :] = v
        total = pos0 + T
        o, _ = _ops.attention_forward(q, kk[:, :, :total, :], vv[:, :, :total, :],
                                      causal=True)
        o = np.ascontiguousarray(o.transpose(0, 2, 1, 3)).reshape(B, T, C)
        return _lin(o, P[s + "proj"])

    def forward(self, ids: List[int], cache: Optional[KVCache] = None,
                pos0: int = 0):
        """返回 (logits:(T,V), cache)。ids 为本次新增 token；pos0 为其绝对位置起点。"""
        P = self.p
        idx = np.asarray(ids, dtype=np.int64).reshape(1, -1)
        T = idx.shape[1]
        if pos0 + T > self.ctx:
            raise ValueError(f"位置越界：{pos0}+{T} > {self.ctx}")
        if cache is None:
            cache = KVCache(self.n_layer, self.n_head, self.ctx, self.dh)
        pos = np.arange(pos0, pos0 + T, dtype=np.int64)
        x = P["token_emb"][idx] + P["pos_emb"][pos][None, :, :]
        for i in range(self.n_layer):
            s = f"blocks.{i}."
            h = _ln(x, P[s + "ln1.weight"], P[s + "ln1.bias"])
            x = x + self._attn(h, i, cache, pos0)
            h2 = _ln(x, P[s + "ln2.weight"], P[s + "ln2.bias"])
            f = _lin(h2, P[s + "fc"], P[s + "fc.b"])
            f = _lin(gelu_exact(f), P[s + "ffp"], P[s + "ffp.b"])
            x = x + f
        cache.len = pos0 + T
        x = _ln(x, P["ln_f.weight"], P["ln_f.bias"])
        logits = _lin(x, P["token_emb"].T)  # 权重绑定：head = token_emb^T
        return logits, cache

    # ----------------------------- 采样 -----------------------------
    @staticmethod
    def _sample(logits: np.ndarray, temperature=0.8, top_k=40, top_p=0.95):
        z = logits.astype(np.float64)
        if temperature <= 1e-5:
            return int(np.argmax(z))
        z = z / temperature
        if top_k and top_k > 0 and top_k < z.size:
            thr = np.partition(z, -top_k)[-top_k]
            z = np.where(z < thr, -np.inf, z)
        m = z.max()
        e = np.exp(z - m)
        p = e / e.sum()
        if top_p and 0 < top_p < 1.0:
            order = np.argsort(-p)
            csum = np.cumsum(p[order])
            keep = order[:int(np.searchsorted(csum, top_p) + 1)]
            mask = np.zeros_like(p, dtype=bool)
            mask[keep] = True
            p = np.where(mask, p, 0.0)
            p = p / p.sum()
        return int(np.random.choice(p.size, p=p))

    # ---------------------------- 生成（流式） ----------------------------
    def generate(self, prompt: str, max_new_tokens=80, temperature=0.8,
                 top_k=40, top_p=0.95,
                 on_token: Optional[Callable[[int, str], None]] = None,
                 stop_strings: Tuple[str, ...] = ("\nUser:", "\nuser:"),
                 seed: Optional[int] = None):
        """返回 dict：text / tokens / prefill_ms / first_token_ms / tok_per_s 等。"""
        if seed is not None:
            np.random.seed(seed)
        ids = self.tok.encode(prompt)
        if len(ids) > self.ctx - 2:
            ids = ids[-(self.ctx - 2):]
        window = list(ids)
        t0 = time.perf_counter()
        logits, cache = self.forward(window, None, 0)
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        n_past = len(window)
        t_gen0 = time.perf_counter()
        out_ids: List[int] = []
        t_first = None
        text_acc = ""
        for _ in range(max_new_tokens):
            if n_past >= self.ctx:  # 窗口滑出最旧 token 并重建 cache
                window = window[-(self.ctx // 2):]
                logits, cache = self.forward(window, None, 0)
                n_past = len(window)
            tok = self._sample(logits[0, -1], temperature, top_k, top_p)
            if t_first is None:
                t_first = (time.perf_counter() - t_gen0) * 1000.0
            out_ids.append(tok)
            piece = self.tok.decode([tok], strip_prefix_space=False)
            text_acc += piece
            if on_token is not None:
                on_token(tok, piece)
            if tok == self.eos_id:
                break
            if any(st in text_acc for st in stop_strings):
                break
            window.append(tok)
            logits, cache = self.forward([tok], cache, n_past)
            n_past += 1
        gen_s = time.perf_counter() - t_gen0
        n_tok = max(1, len(out_ids))
        for st in stop_strings:
            if st in text_acc:
                text_acc = text_acc.split(st)[0]
        self.peak_mem = _peak_wset_mb()
        return {
            "text": text_acc,
            "tokens": out_ids,
            "n_tokens": len(out_ids),
            "prompt_tokens": len(ids),
            "prefill_ms": prefill_ms,
            "first_token_ms": t_first if t_first is not None else -1.0,
            "gen_seconds": gen_s,
            "tok_per_s": n_tok / gen_s if gen_s > 0 else 0.0,
            "peak_mem_mb": self.peak_mem,
        }


def _peak_wset_mb() -> float:
    """Windows 峰值工作集（MB）；非 Windows 返回 0。"""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        h = k32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0
