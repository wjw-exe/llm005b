# -*- coding: utf-8 -*-
"""GPT 风格 decoder-only 语言模型（numpy 手写前向 / 反向）。

结构：Token Embedding + 可学习位置嵌入 → N × [LayerNorm → 因果自注意力 → 残差
→ LayerNorm → FFN(GELU) → 残差] → 最终 LayerNorm → LM Head（与词嵌入权重共享）
"""
import json
import numpy as np

import ops
from ops import DTYPE


def _split_heads(x, B, T, H, Dh):
    return np.ascontiguousarray(x.reshape(B, T, H, Dh).transpose(0, 2, 1, 3))


def _merge_heads(x, B, T, H, Dh):
    return np.ascontiguousarray(x.transpose(0, 2, 1, 3)).reshape(B, T, H * Dh)


class GPT:
    def __init__(self, cfg, seed=20261121):
        self.cfg = dict(cfg)
        self.V = int(cfg["vocab_size"])
        self.C = int(cfg["d_model"])
        self.L = int(cfg["n_layer"])
        self.H = int(cfg["n_head"])
        self.Dh = self.C // self.H
        self.F = int(cfg.get("ffn_dim", 4 * self.C))
        self.ctx = int(cfg["ctx"])
        self.tied = bool(cfg.get("tied", True))
        self.rng = np.random.default_rng(seed)
        self._init_params()

    # ------------------------------------------------------------------ 参数
    def _init_params(self):
        C, F, V, L = self.C, self.F, self.V, self.L
        r = self.rng
        std = 0.02
        res_std = std / np.sqrt(2.0 * L)
        p = {}
        p["wte"] = (r.normal(0, std, (V, C))).astype(DTYPE)
        p["wpe"] = (r.normal(0, std, (self.ctx, C))).astype(DTYPE)
        for i in range(L):
            k = "l%d." % i
            p[k + "ln1_g"] = np.ones(C, DTYPE)
            p[k + "ln1_b"] = np.zeros(C, DTYPE)
            for nm, bn, so, sd in (("Wq", "qb", C, res_std), ("Wk", "kb", C, res_std),
                                   ("Wv", "vb", C, res_std), ("Wo", "ob", C, res_std)):
                p[k + nm] = (r.normal(0, sd, (C, so))).astype(DTYPE)
                p[k + bn] = np.zeros(so, DTYPE)
            p[k + "ln2_g"] = np.ones(C, DTYPE)
            p[k + "ln2_b"] = np.zeros(C, DTYPE)
            p[k + "W1"] = (r.normal(0, std, (C, F))).astype(DTYPE)
            p[k + "b1"] = np.zeros(F, DTYPE)
            p[k + "W2"] = (r.normal(0, res_std, (F, C))).astype(DTYPE)
            p[k + "b2"] = np.zeros(C, DTYPE)
        p["lnf_g"] = np.ones(C, DTYPE)
        p["lnf_b"] = np.zeros(C, DTYPE)
        if not self.tied:
            p["head"] = (r.normal(0, std, (C, V))).astype(DTYPE)
        self.params = p

    def param_shapes(self):
        return {k: v.shape for k, v in self.params.items()}

    def num_params(self):
        n = sum(int(v.size) for v in self.params.values())
        if self.tied:
            n += 0  # 权重共享，LM Head 不额外占参数
        return n

    def zero_grads(self):
        """梯度缓冲只分配一次，后续原地清零（避免每步 malloc 数百 MB）。"""
        g = getattr(self, "grads", None)
        if g is not None and len(g) == len(self.params):
            for v in g.values():
                v.fill(0.0)
            return
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    # ---------------------------------------------------------------- 前向
    def forward(self, idx, targets=None):
        """训练前向。idx:(B,T) int -> logits:(B,T,V)，并缓存反向所需中间量。"""
        p = self.params
        B, T = idx.shape
        if T > self.ctx:
            raise ValueError("序列长度 %d 超过上下文 %d" % (T, self.ctx))
        x = p["wte"][idx] + p["wpe"][:T]
        layers = []
        for i in range(self.L):
            k = "l%d." % i
            x0 = x
            h1, c_ln1 = ops.layernorm_forward(x0, p[k + "ln1_g"], p[k + "ln1_b"])
            q, c_q = ops.linear_forward(h1, p[k + "Wq"], p[k + "qb"])
            kk, c_k = ops.linear_forward(h1, p[k + "Wk"], p[k + "kb"])
            v, c_v = ops.linear_forward(h1, p[k + "Wv"], p[k + "vb"])
            qh = _split_heads(q, B, T, self.H, self.Dh)
            kh = _split_heads(kk, B, T, self.H, self.Dh)
            vh = _split_heads(v, B, T, self.H, self.Dh)
            oh, c_attn = ops.attention_forward(qh, kh, vh, causal=True)
            om = _merge_heads(oh, B, T, self.H, self.Dh)
            ao, c_o = ops.linear_forward(om, p[k + "Wo"], p[k + "ob"])
            x1 = x0 + ao
            h2, c_ln2 = ops.layernorm_forward(x1, p[k + "ln2_g"], p[k + "ln2_b"])
            u, c_u1 = ops.linear_forward(h2, p[k + "W1"], p[k + "b1"])
            g, c_gelu = ops.gelu_forward(u)
            w, c_u2 = ops.linear_forward(g, p[k + "W2"], p[k + "b2"])
            x = x1 + w
            layers.append(dict(ln1=c_ln1, h1=h1, cq=c_q, ck=c_k, cv=c_v, attn=c_attn,
                               co=c_o, x0=x0, x1=x1, ln2=c_ln2, h2=h2, u1=c_u1,
                               gelu=c_gelu, u2=c_u2))
        xf, c_lnf = ops.layernorm_forward(x, p["lnf_g"], p["lnf_b"])
        if self.tied:
            logits = xf @ p["wte"].T
        else:
            logits = xf @ p["head"]
        self.cache = dict(idx=idx, layers=layers, xf=xf, lnf=c_lnf, B=B, T=T)
        if targets is None:
            return logits
        loss, c_ce = ops.cross_entropy_forward(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        self.cache["ce"] = c_ce
        return loss

    # ---------------------------------------------------------------- 反向
    def backward(self, dlogits=None):
        p, g, c = self.params, self.grads, self.cache
        B, T = c["B"], c["T"]
        if dlogits is None:
            dlogits = ops.cross_entropy_backward(DTYPE(1.0), c["ce"])
        dlogits = dlogits.reshape(B, T, -1)
        xf = c["xf"]
        d2 = dlogits.reshape(-1, dlogits.shape[-1])
        xf2 = xf.reshape(-1, self.C)
        if self.tied:
            g["wte"] += d2.T @ xf2
            dxf = dlogits @ p["wte"]
        else:
            g["head"] += d2.T @ xf2
            dxf = dlogits @ p["head"].T
        dx, dg, db = ops.layernorm_backward(dxf, c["lnf"])
        g["lnf_g"] += dg
        g["lnf_b"] += db
        for i in range(self.L - 1, -1, -1):
            k = "l%d." % i
            lc = c["layers"][i]
            # 残差 2
            d_res2, dw = ops.residual_backward(dx, None)
            dg2 = ops.linear_backward(dw, lc["u2"])
            g[k + "W2"] += dg2[1]
            g[k + "b2"] += dg2[2]
            dgel = ops.gelu_backward(dg2[0], lc["gelu"])
            du = ops.linear_backward(dgel, lc["u1"])
            g[k + "W1"] += du[1]
            g[k + "b1"] += du[2]
            dln2 = ops.layernorm_backward(du[0], lc["ln2"])
            g[k + "ln2_g"] += dln2[1]
            g[k + "ln2_b"] += dln2[2]
            dx1 = d_res2 + dln2[0]
            # 残差 1
            d_res1, dw1 = ops.residual_backward(dx1, None)
            dao = ops.linear_backward(dw1, lc["co"])
            g[k + "Wo"] += dao[1]
            g[k + "ob"] += dao[2]
            doh = _split_heads(dao[0], B, T, self.H, self.Dh)
            dqh, dkh, dvh = ops.attention_backward(doh, lc["attn"])
            dqm = _merge_heads(dqh, B, T, self.H, self.Dh)
            dkm = _merge_heads(dkh, B, T, self.H, self.Dh)
            dvm = _merge_heads(dvh, B, T, self.H, self.Dh)
            dq = ops.linear_backward(dqm, lc["cq"])
            dk_ = ops.linear_backward(dkm, lc["ck"])
            dv_ = ops.linear_backward(dvm, lc["cv"])
            g[k + "Wq"] += dq[1]
            g[k + "qb"] += dq[2]
            g[k + "Wk"] += dk_[1]
            g[k + "kb"] += dk_[2]
            g[k + "Wv"] += dv_[1]
            g[k + "vb"] += dv_[2]
            dh1 = dq[0] + dk_[0] + dv_[0]
            dln1 = ops.layernorm_backward(dh1, lc["ln1"])
            g[k + "ln1_g"] += dln1[1]
            g[k + "ln1_b"] += dln1[2]
            dx = d_res1 + dln1[0]
        # 输入侧：词嵌入 + 位置嵌入
        g["wte"] += ops.embedding_backward(dx, (c["idx"], p["wte"].shape), self.V)
        g["wpe"][:T] += dx.sum(axis=0)

    # ------------------------------------------------------- 推理（KV Cache）
    def new_kv(self):
        return [dict(k=None, v=None) for _ in range(self.L)]

    def forward_infer(self, idx, kv):
        """增量推理：idx 只需传入未处理过的新 token。返回最后一步 logits 与更新后的 kv。"""
        p = self.params
        B, T = idx.shape
        pos0 = 0 if kv[0]["k"] is None else kv[0]["k"].shape[2]
        x = p["wte"][idx] + p["wpe"][pos0:pos0 + T]
        for i in range(self.L):
            k = "l%d." % i
            h1, _ = ops.layernorm_forward(x, p[k + "ln1_g"], p[k + "ln1_b"])
            q = h1 @ p[k + "Wq"] + p[k + "qb"]
            kk = h1 @ p[k + "Wk"] + p[k + "kb"]
            v = h1 @ p[k + "Wv"] + p[k + "vb"]
            qh = _split_heads(q, B, T, self.H, self.Dh)
            kh = _split_heads(kk, B, T, self.H, self.Dh)
            vh = _split_heads(v, B, T, self.H, self.Dh)
            if kv[i]["k"] is not None:
                kh_all = np.concatenate([kv[i]["k"], kh], axis=2)
                vh_all = np.concatenate([kv[i]["v"], vh], axis=2)
            else:
                kh_all, vh_all = kh, vh
            kv[i]["k"], kv[i]["v"] = kh_all, vh_all
            oh, _ = ops.attention_forward(qh, kh_all, vh_all, causal=True)
            om = _merge_heads(oh, B, T, self.H, self.Dh)
            x = x + (om @ p[k + "Wo"] + p[k + "ob"])
            h2, _ = ops.layernorm_forward(x, p[k + "ln2_g"], p[k + "ln2_b"])
            u = h2 @ p[k + "W1"] + p[k + "b1"]
            u3 = u * u * u
            gg = DTYPE(0.5) * u * (1.0 + np.tanh(np.sqrt(DTYPE(2.0 / np.pi)) * (u + DTYPE(0.044715) * u3)))
            x = x + (gg @ p[k + "W2"] + p[k + "b2"])
        xf, _ = ops.layernorm_forward(x, p["lnf_g"], p["lnf_b"])
        if self.tied:
            logits = xf @ p["wte"].T
        else:
            logits = xf @ p["head"]
        return logits[:, -1, :], kv

    # ----------------------------------------------------------- 存取
    def state_arrays(self, prefix=""):
        return {prefix + k: v for k, v in self.params.items()}

    def load_state(self, arrays):
        for k in self.params:
            if k in arrays:
                v = arrays[k]
                if v.shape != self.params[k].shape:
                    raise ValueError("参数形状不匹配：%s 期望 %s 得到 %s"
                                     % (k, self.params[k].shape, v.shape))
                self.params[k] = v.astype(DTYPE, copy=False)


def default_config():
    return {
        "vocab_size": 16000,
        "d_model": 512,
        "n_layer": 12,
        "n_head": 8,
        "ffn_dim": 2048,
        "ctx": 256,
        "tied": True,
        "dropout": 0.0,
    }
