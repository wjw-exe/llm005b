# -*- coding: utf-8 -*-
"""数值梯度校验：用中心差分验证手写反向传播的正确性。

被测算子：Linear、LayerNorm、Softmax、Causal Attention（另附 GELU / CrossEntropy）
指标：相对误差 rel = |g_analytic - g_numeric| / (|g_analytic| + |g_numeric| + 1e-12)
通过标准：所有参量的最大相对误差 < 1e-5
"""
import sys
import numpy as np

import ops

# 校验全程用 float64，避免 float32 舍入把差分误差放大到 1e-3 量级
ops.DTYPE = np.float64
ops._GELU_K = np.sqrt(2.0 / np.pi)
ops._GELU_C = 0.044715
ops.NEG = -1e30

EPS = 1e-6


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12)))


def _num_grad(f, arr, dout, idx=None):
    """对 f(arr)->y 做中心差分，返回与 arr 同形的数值梯度（仅对 idx 指定位置）。"""
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + EPS
        fp = np.sum(f(arr) * dout)
        arr[i] = old - EPS
        fm = np.sum(f(arr) * dout)
        arr[i] = old
        g[i] = (fp - fm) / (2 * EPS)
        it.iternext()
    return g


def check_linear(rng):
    x = rng.normal(size=(2, 3, 5))
    W = rng.normal(size=(5, 4))
    b = rng.normal(size=(4,))
    dout = rng.normal(size=(2, 3, 4))
    y, cache = ops.linear_forward(x, W, b)
    dx, dW, db = ops.linear_backward(dout, cache)
    ndx = _num_grad(lambda a: ops.linear_forward(a, W, b)[0], x, dout)
    ndW = _num_grad(lambda a: ops.linear_forward(x, a, b)[0], W, dout)
    ndb = _num_grad(lambda a: ops.linear_forward(x, W, a)[0], b, dout)
    return {"dx": _rel(dx, ndx), "dW": _rel(dW, ndW), "db": _rel(db, ndb)}


def check_layernorm(rng):
    x = rng.normal(size=(2, 4, 8)) * 1.3 + 0.4
    g = rng.normal(size=(8,))
    b = rng.normal(size=(8,))
    dout = rng.normal(size=(2, 4, 8))
    y, cache = ops.layernorm_forward(x, g, b)
    dx, dg, db = ops.layernorm_backward(dout, cache)
    ndx = _num_grad(lambda a: ops.layernorm_forward(a, g, b)[0], x, dout)
    ndg = _num_grad(lambda a: ops.layernorm_forward(x, a, b)[0], g, dout)
    ndb = _num_grad(lambda a: ops.layernorm_forward(x, g, a)[0], b, dout)
    return {"dx": _rel(dx, ndx), "dg": _rel(dg, ndg), "db": _rel(db, ndb)}


def check_softmax(rng):
    x = rng.normal(size=(3, 7)) * 2.0
    dout = rng.normal(size=(3, 7))
    y, _ = ops.softmax_forward(x, axis=-1)
    dx = ops.softmax_backward(dout, y, axis=-1)
    ndx = _num_grad(lambda a: ops.softmax_forward(a, axis=-1)[0], x, dout)
    return {"dx": _rel(dx, ndx)}


def check_gelu(rng):
    x = rng.normal(size=(2, 6))
    dout = rng.normal(size=(2, 6))
    y, cache = ops.gelu_forward(x)
    dx = ops.gelu_backward(dout, cache)
    ndx = _num_grad(lambda a: ops.gelu_forward(a)[0], x, dout)
    return {"dx": _rel(dx, ndx)}


def check_attention(rng):
    B, H, T, Dh = 2, 2, 4, 4
    q = rng.normal(size=(B, H, T, Dh))
    k = rng.normal(size=(B, H, T, Dh))
    v = rng.normal(size=(B, H, T, Dh))
    dout = rng.normal(size=(B, H, T, Dh))
    y, cache = ops.attention_forward(q, k, v, causal=True)
    dq, dk, dv = ops.attention_backward(dout, cache)
    ndq = _num_grad(lambda a: ops.attention_forward(a, k, v, causal=True)[0], q, dout)
    ndk = _num_grad(lambda a: ops.attention_forward(q, a, v, causal=True)[0], k, dout)
    ndv = _num_grad(lambda a: ops.attention_forward(q, k, a, causal=True)[0], v, dout)
    return {"dq": _rel(dq, ndq), "dk": _rel(dk, ndk), "dv": _rel(dv, ndv)}


def check_cross_entropy(rng):
    N, V = 5, 9
    logits = rng.normal(size=(N, V))
    target = rng.integers(0, V, size=(N,))
    loss, cache = ops.cross_entropy_forward(logits, target)
    dlogits = ops.cross_entropy_backward(1.0, cache)
    num = np.zeros_like(logits)
    for i in range(N):
        for j in range(V):
            old = logits[i, j]
            logits[i, j] = old + EPS
            lp = ops.cross_entropy_forward(logits, target)[0]
            logits[i, j] = old - EPS
            lm = ops.cross_entropy_forward(logits, target)[0]
            logits[i, j] = old
            num[i, j] = (lp - lm) / (2 * EPS)
    return {"dlogits": _rel(dlogits, num)}


def main():
    rng = np.random.default_rng(7)
    cases = [
        ("Linear (forward: y=xW+b)", check_linear),
        ("LayerNorm (forward: 归一化+仿射)", check_layernorm),
        ("Softmax (forward: 归一化概率)", check_softmax),
        ("Causal Attention (forward: 因果掩码+qk^T/√d·v)", check_attention),
        ("GELU (forward: tanh 近似)", check_gelu),
        ("CrossEntropy (forward: fused softmax+NLL)", check_cross_entropy),
    ]
    worst = 0.0
    print("=" * 74)
    print("算子数值梯度校验（中心差分，eps=%.0e，float64）" % EPS)
    print("=" * 74)
    fail = False
    for name, fn in cases:
        res = fn(rng)
        m = max(res.values())
        worst = max(worst, m)
        ok = m < 1e-5
        fail = fail or not ok
        detail = ", ".join("%s=%.3e" % (k, v) for k, v in res.items())
        print("%-46s %s  max_rel=%.3e  [%s]" % (name, detail, m, "PASS" if ok else "FAIL"))
    print("-" * 74)
    print("最大相对误差 = %.3e ；判定标准 < 1e-5 ；总体结果：%s"
          % (worst, "全部通过 (PASS)" if not fail else "存在失败 (FAIL)"))
    print("=" * 74)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
