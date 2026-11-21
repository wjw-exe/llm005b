# -*- coding: utf-8 -*-
"""llm005b 命令行对话前端（纯 numpy 手写推理，无深度学习框架）。

用法：
    python chat.py                     # 自动加载 ckpt/final.npz 或 latest.npz
    python chat.py --ckpt ckpt/step_300.npz --temp 0.8 --topk 40

斜杠命令：
    /reset  清空多轮上下文        /temp 0.7   设置温度
    /topk 40   设置 top-k         /topp 0.9   设置 top-p
    /stats  查看运行统计          /exit       退出（Ctrl+C 亦可）
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bp.bpe import BPETokenizer  # noqa: E402
from model import GPT, default_config  # noqa: E402

STOP_PATTERNS = ("\nUser:", "\nUser：", "\nuser:", "\nAssistant:", "\nAssistant：")


def human(n):
    if n >= 1e6:
        return "%.2fM" % (n / 1e6)
    if n >= 1e3:
        return "%.1fk" % (n / 1e3)
    return "%d" % n


def rss_mb():
    """返回（当前工作集 MB, 峰值工作集 MB）；失败返回 (-1, -1)。"""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        psapi = ctypes.WinDLL("psapi")
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        psapi.GetProcessMemoryInfo.restype = wt.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return -1.0, -1.0
        return pmc.WorkingSetSize / 1048576.0, pmc.PeakWorkingSetSize / 1048576.0
    except Exception:
        return -1.0, -1.0


def find_ckpt(explicit):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    cdir = os.path.join(ROOT, "ckpt")
    # 0) 指针文件 latest.json（训练器写入的权威指针）
    pj = os.path.join(cdir, "latest.json")
    if os.path.exists(pj):
        try:
            with open(pj, encoding="utf-8") as fh:
                info = json.load(fh)
            cand = os.path.join(cdir, info.get("file", ""))
            if os.path.exists(cand):
                return cand
        except Exception:
            pass
    cands = []
    for name in ("final.npz", "latest.npz"):
        p = os.path.join(ROOT, "ckpt", name)
        if os.path.exists(p):
            cands.append((os.path.getmtime(p), p))
    if not cands:
        for f in os.listdir(os.path.join(ROOT, "ckpt")) if os.path.isdir(os.path.join(ROOT, "ckpt")) else []:
            if f.endswith(".npz"):
                p = os.path.join(ROOT, "ckpt", f)
                cands.append((os.path.getmtime(p), p))
    if not cands:
        return None
    cands.sort()
    return cands[-1][1]


def sample_from(logits, temp, top_k, top_p, rng):
    """手写采样：温度 / top-k / top-p（nucleus）+ 多项分布。"""
    z = logits.astype(np.float64).copy()
    if temp <= 1e-5:
        return int(np.argmax(z))
    z /= temp
    if top_k and top_k > 0 and top_k < z.size:
        kth = np.partition(z, -top_k)[-top_k]
        z[z < kth] = -np.inf
    if top_p and 0.0 < top_p < 1.0:
        order = np.argsort(-z)
        sz = z[order]
        m = np.isfinite(sz)
        sz = sz[m]
        idx = order[m]
        e = np.exp(sz - sz.max())
        cdf = np.cumsum(e / e.sum())
        keep = int(np.searchsorted(cdf, top_p) + 1)
        keep = max(1, min(keep, idx.size))
        z.fill(-np.inf)
        z[idx[:keep]] = logits[idx[:keep]].astype(np.float64) / temp
    e = np.exp(z - np.max(z))
    prob = e / e.sum()
    return int(rng.choice(prob.size, p=prob))


class Chat(object):
    def __init__(self, ckpt_path, temp=0.85, top_k=40, top_p=0.9, max_new=96):
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.max_new = max_new
        self.rng = np.random.default_rng(1234)
        self.history = []
        self.total_gen_tokens = 0
        self.total_gen_time = 0.0
        self.turns = 0
        self.ckpt_path = ckpt_path

        tok_path = os.path.join(ROOT, "bp", "tokenizer.json")
        if not os.path.exists(tok_path):
            raise SystemExit("找不到分词器 %s，请先运行 prepare.py 或在菜单里准备语料。" % tok_path)
        self.tk = BPETokenizer.load(tok_path)

        meta = {}
        ck = np.load(ckpt_path, allow_pickle=True)
        keys = set(ck.files)
        if "config" in keys:
            raw = ck["config"].item()
            meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
        elif "cfg" in keys:
            raw = ck["cfg"].item()
            meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
        cfg = default_config()
        cfg.update({k: v for k, v in meta.items() if k in cfg})
        self.ctx = int(cfg["ctx"])
        self.model = GPT(cfg)
        arrs = {k: ck[k] for k in keys if k not in ("config", "cfg", "step", "loss", "tokens", "best", "meta")}
        self.model.load_state(arrs)
        self.step = int(ck["step"]) if "step" in keys else -1
        self.loss = float(ck["loss"]) if "loss" in keys else float("nan")
        self.params = self.model.num_params()

    # ------------------------------------------------------------ prompt
    def build_ids(self, user_text):
        """保留最近若干轮，保证 prompt + max_new 不超过 ctx。"""
        reserve = min(self.max_new, self.ctx // 2)
        budget = self.ctx - reserve
        hist = list(self.history)
        while True:
            txt = "".join("User: %s\nAssistant: %s\n\n" % (u, a) for u, a in hist)
            txt += "User: %s\nAssistant: " % user_text
            ids = self.tk.encode(txt)
            if len(ids) <= budget or not hist:
                break
            hist.pop(0)
        if len(ids) > budget:  # 单轮就超长：从左侧硬截断
            ids = ids[-budget:]
        return ids, len(hist)

    def generate(self, prompt_ids):
        kv = self.model.new_kv()
        t0 = time.time()
        x = np.array([prompt_ids], dtype=np.int64)
        logits, kv = self.model.forward_infer(x, kv)
        first = time.time() - t0
        used = len(prompt_ids)

        raw = b""
        text = ""
        shown = 0
        n_new = 0
        tgen = time.time()
        out_tokens = []
        while n_new < self.max_new and used < self.ctx:
            tok = sample_from(logits[0], self.temp, self.top_k, self.top_p, self.rng)
            if tok < 2:  # 特殊 token 不输出
                n_new += 1
                break
            out_tokens.append(tok)
            raw += self.tk.token_bytes(tok)
            n_new += 1
            used += 1
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = None  # 半个汉字，等下一个 token
            if text is not None:
                new = text[shown:]
                if new:
                    sys.stdout.write(new)
                    sys.stdout.flush()
                    shown = len(text)
                hit = [p for p in STOP_PATTERNS if p in text]
                if hit:
                    break
            logits, kv = self.model.forward_infer(np.array([[tok]], dtype=np.int64), kv)
        dt = time.time() - tgen
        try:
            full = raw.decode("utf-8", errors="ignore")
        except Exception:
            full = ""
        for p in STOP_PATTERNS:
            if p in full:
                full = full.split(p)[0]
        self.total_gen_tokens += n_new
        self.total_gen_time += dt
        return full.strip(), n_new, dt, first, used

    # -------------------------------------------------------------- loop
    def banner(self):
        rss, peak = rss_mb()
        print("=" * 62)
        print(" llm005b 对话助手  |  纯 numpy 手写算子，无任何深度学习框架")
        print(" 权重: %s" % os.path.relpath(self.ckpt_path, ROOT))
        print(" 结构: d_model=%d  layers=%d  heads=%d  ctx=%d  vocab=%d"
              % (self.model.C, self.model.L, self.model.H, self.ctx, self.model.V))
        print(" 参数: %s  训练步: %d  最近 loss: %.4f" % (human(self.params), self.step, self.loss))
        print(" 采样: temp=%.2f top_k=%d top_p=%.2f max_new=%d" % (self.temp, self.top_k, self.top_p, self.max_new))
        if rss > 0:
            print(" 内存: 当前 %.0f MB / 峰值 %.0f MB" % (rss, peak))
        print(" 输入问题后回车；/help 查看命令；Ctrl+C 退出")
        print("=" * 62)

    def stats(self):
        rss, peak = rss_mb()
        speed = (self.total_gen_tokens / self.total_gen_time) if self.total_gen_time > 0 else 0.0
        print("---- 运行统计 ----")
        print(" 对话轮数        : %d" % self.turns)
        print(" 已生成 token    : %d" % self.total_gen_tokens)
        print(" 平均生成速度    : %.2f token/s（%.1f ms/token）"
              % (speed, (1000.0 / speed) if speed > 0 else 0))
        print(" 上下文上限      : %d token" % self.ctx)
        print(" 当前上下文占用  : %d token" % self.current_len())
        print(" 参数总量        : %s" % human(self.params))
        if rss > 0:
            print(" 内存峰值        : %.0f MB" % peak)

    def current_len(self):
        txt = "".join("User: %s\nAssistant: %s\n\n" % (u, a) for u, a in self.history)
        return len(self.tk.encode(txt)) if txt else 0

    def run(self):
        self.banner()
        while True:
            try:
                line = input("\nYou> ").strip()
            except KeyboardInterrupt:
                print("\n已退出。")
                self.stats()
                return 0
            except EOFError:
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                cmd = line.split()
                name = cmd[0].lower()
                arg = cmd[1] if len(cmd) > 1 else ""
                if name in ("/exit", "/quit", "/q"):
                    self.stats()
                    return 0
                elif name in ("/reset", "/clear"):
                    self.history = []
                    print("上下文已清空。")
                elif name == "/temp":
                    try:
                        self.temp = float(arg)
                        print("温度 = %.2f" % self.temp)
                    except ValueError:
                        print("用法：/temp 0.8")
                elif name == "/topk":
                    try:
                        self.top_k = int(arg)
                        print("top_k = %d" % self.top_k)
                    except ValueError:
                        print("用法：/topk 40")
                elif name == "/topp":
                    try:
                        self.top_p = float(arg)
                        print("top_p = %.2f" % self.top_p)
                    except ValueError:
                        print("用法：/topp 0.9")
                elif name == "/model":
                    if arg in ("", "scratch", "self", "old"):
                        print("当前模型：工程自训练模型（ckpt/final.npz）")
                        print("可用：/model instruct 切换到 0.05B 指令微调模型（公开权重）")
                    elif arg in ("instruct", "0.05b", "new"):
                        print("正在切换到 0.05B 指令微调模型...")
                        return "instruct"
                    else:
                        print("用法：/model [instruct|scratch]")
                elif name == "/stats":
                    self.stats()
                elif name == "/help":
                    print("命令：/reset /temp <v> /topk <n> /topp <p> "
                          "/model [instruct|scratch] /stats /exit")
                else:
                    print("未知命令，/help 查看。")
                continue

            prompt_ids, kept = self.build_ids(line)
            if kept < len(self.history):
                print("（上下文超限，已自动丢弃最旧的 %d 轮）" % (len(self.history) - kept))
            print("Assistant> ", end="", flush=True)
            try:
                answer, n_new, dt, first, used = self.generate(prompt_ids)
            except KeyboardInterrupt:
                print("\n[已中断本次生成]")
                continue
            if not answer:
                print("（模型只输出了空内容，可尝试 /temp 0.7 或 /reset）")
                answer = ""
            speed = n_new / dt if dt > 0 else 0.0
            print("\n[%d tok / %.2fs = %.2f tok/s | 首 token %.2fs | 上下文 %d/%d]"
                  % (n_new, dt, speed, first, used, self.ctx))
            self.history.append((line, answer))
            self.turns += 1


def _build_scratch(args):
    ck = find_ckpt(args.ckpt)
    if ck is None:
        ckdir = os.path.join(ROOT, "ckpt")
        print("=" * 62)
        print(" 未找到可用的 checkpoint。")
        print(" 期望位置：%s" % ckdir)
        print(" 请先训练出权重，例如：")
        print("     python train.py --steps 400 --batch 8")
        print("   或直接双击 start.bat 选择「2 训练模型」。")
        print("=" * 62)
        return None, 2
    try:
        app = Chat(ck, temp=(args.temp if args.temp is not None else 0.85), top_k=args.topk, top_p=args.topp, max_new=args.max_new)
    except SystemExit:
        raise
    except Exception as e:
        print("加载 checkpoint 失败：%s: %s" % (type(e).__name__, e))
        print("建议检查该 .npz 是否完整（config/step/参数数组齐备），或改用 ckpt/latest.npz。")
        return None, 3
    return app, 0



def _build_instruct(args):
    try:
        from inf.instruct_chat import InstructChat
    except Exception as e:
        print("加载指令模型模块失败：%s: %s" % (type(e).__name__, e))
        return None
    t = args.temp if args.temp is not None else 0.5
    try:
        return InstructChat(temp=t, top_k=args.topk, top_p=args.topp,
                            max_new=args.max_new, mode=args.mode,
                            translate=bool(getattr(args, "translate", False)),
                            provider=getattr(args, "provider", "") or "",
                            lib_endpoint=getattr(args, "libre_endpoint", "") or "",
                            lib_key=getattr(args, "libre_key", "") or "")
    except SystemExit:
        return None


def main():
    ap = argparse.ArgumentParser(description="llm005b 对话前端（手写 numpy 推理）")
    ap.add_argument("--model", default="instruct", choices=["instruct", "scratch"],
                    help="instruct=0.05B 公开指令微调权重（默认）；scratch=工程自训练模型")
    ap.add_argument("--mode", default="single", choices=["single", "multi"],
                    help="上下文模式：single 仅当前问题（贴合模型单轮训练分布）；multi 拼接历史")
    ap.add_argument("--ckpt", default="", help="checkpoint 路径（默认 ckpt/final.npz 或 latest.npz）")
    ap.add_argument("--temp", type=float, default=None,
                    help="采样温度（instruct 默认 0.5，scratch 默认 0.85）")
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--topp", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--translate", action="store_true",
                    help="开启公共翻译 API 中文增强：中文输入→译英→模型→英文回答→回译中文"
                         "（需联网；不开启则纯英文直连）")
    ap.add_argument("--provider", default="",
                    help="翻译提供方链（顺序回退），默认 mymemory；"
                         "可写 mymemory,libretranslate 或 mymemory+libretranslate")
    ap.add_argument("--libre-endpoint", default="",
                    help="LibreTranslate 端点，如 http://127.0.0.1:5000（默认不启用）")
    ap.add_argument("--libre-key", default="",
                    help="LibreTranslate api key（公共实例通常需要）")
    args = ap.parse_args()

    if args.translate and args.model == "scratch":
        print("（提示：中文增强仅接入 0.05B 指令微调模型；自训练模型仍按原纯英文通路运行）")


    kind = args.model
    while True:
        if kind == "instruct":
            app = _build_instruct(args)
            if app is None:
                return 2
        else:
            app, code = _build_scratch(args)
            if app is None:
                return code
        nxt = app.run()
        if nxt in ("instruct", "scratch"):
            kind = nxt
            print("=" * 62)
            print(" 已切换模型，正在加载 %s ..." % (
                "0.05B 指令微调模型" if kind == "instruct" else "工程自训练模型"))
            print("=" * 62)
            continue
        return nxt if isinstance(nxt, int) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(0)
