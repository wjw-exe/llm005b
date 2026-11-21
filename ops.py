# -*- coding: utf-8 -*-
"""手写张量算子库：每个算子成对提供 forward / backward。

约定
----
* 只依赖 numpy 与标准库，全程 float32。
* 张量布局为 batch 优先：(B, T, C)；注意力内部使用 (B, H, T, Dh)。
* 所有 backward 返回的梯度与对应参数/输入形状严格一致，便于有限差分校验。
"""
import numpy as np

DTYPE = np.float32
NEG = DTYPE(-1e30)


# ================================ Linear ================================
def linear_forward(x, W, b=None):
    """y = x @ W + b ；x:(..., Cin) W:(Cin, Cout)"""
    xs = x.shape
    x2 = np.ascontiguousarray(x.reshape(-1, xs[-1]))
    y = x2 @ W
    if b is not None:
        y = y + b
    y = y.reshape(xs[:-1] + (W.shape[1],))
    return y, (x2, W, xs)


def linear_backward(dy, cache, need_dx=True):
    x2, W, xs = cache
    dy2 = np.ascontiguousarray(dy.reshape(-1, dy.shape[-1]))
    dW = x2.T @ dy2
    db = dy2.sum(axis=0).astype(DTYPE, copy=False)
    dx = None
    if need_dx:
        dx = (dy2 @ W.T).reshape(xs)
    return dx, dW, db


# =============================== LayerNorm ==============================
def layernorm_forward(x, g, b, eps=1e-5):
    """标准 LayerNorm（最后一个维度归一化）"""
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    rstd = (1.0 / np.sqrt(var + eps)).astype(DTYPE, copy=False)
    xhat = xc * rstd
    y = xhat * g + b
    return y, (xhat, rstd, g)


def layernorm_backward(dy, cache):
    xhat, rstd, g = cache
    C = xhat.shape[-1]
    flat_dy = dy.reshape(-1, C)
    flat_x = xhat.reshape(-1, C)
    dg = (flat_dy * flat_x).sum(axis=0)
    db = flat_dy.sum(axis=0).astype(DTYPE, copy=False)
    dxhat = dy * g
    m1 = dxhat.mean(axis=-1, keepdims=True)
    m2 = (dxhat * xhat).mean(axis=-1, keepdims=True)
    dx = rstd * (dxhat - m1 - xhat * m2)
    return dx, dg, db


# ================================= GELU =================================
_GELU_K = np.sqrt(DTYPE(2.0 / np.pi))
_GELU_C = DTYPE(0.044715)


def gelu_forward(x):
    """tanh 近似 GELU：0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))"""
    x3 = x * x * x
    u = _GELU_K * (x + _GELU_C * x3)
    t = np.tanh(u)
    y = DTYPE(0.5) * x * (1.0 + t)
    return y, (x, t)


def gelu_backward(dy, cache):
    x, t = cache
    du = _GELU_K * (1.0 + DTYPE(3.0) * _GELU_C * x * x)
    dydx = DTYPE(0.5) * (1.0 + t) + DTYPE(0.5) * x * (1.0 - t * t) * du
    return dy * dydx


# =============================== Softmax ================================
def softmax_forward(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    s = e.sum(axis=axis, keepdims=True)
    y = e / s
    return y.astype(DTYPE, copy=False), y


def softmax_backward(dy, y, axis=-1):
    dot = (dy * y).sum(axis=axis, keepdims=True)
    return y * (dy - dot)


# ============================== Attention ===============================
def _causal_mask(q_len, k_len):
    """返回 True 表示需要屏蔽的位置；支持 q_len=1 的解码场景（无需屏蔽）。"""
    if q_len == 1:
        return None
    i = np.arange(q_len)[:, None]
    j = np.arange(k_len)[None, :]
    return j > (k_len - q_len + i)


def attention_forward(q, k, v, causal=True):
    """q:(B,H,Tq,Dh) k,v:(B,H,Tk,Dh) -> o:(B,H,Tq,Dh)

    带 KV Cache 时调用方把历史 K/V 与当前步拼接到 Tk 维即可，本算子无需改动。
    """
    Dh = q.shape[-1]
    scale = DTYPE(1.0 / np.sqrt(Dh))
    scores = (q @ np.ascontiguousarray(k.transpose(0, 1, 3, 2))) * scale
    mask = _causal_mask(q.shape[2], k.shape[2]) if causal else None
    if mask is not None:
        scores = np.where(mask, NEG, scores)
    p, _ = softmax_forward(scores, axis=-1)
    o = p @ v
    return o, (q, k, v, p, scale, mask)


def attention_backward(do, cache):
    q, k, v, p, scale, mask = cache
    dv = np.ascontiguousarray(p.transpose(0, 1, 3, 2)) @ do
    dp = do @ np.ascontiguousarray(v.transpose(0, 1, 3, 2))
    ds = softmax_backward(dp, p, axis=-1)
    if mask is not None:
        ds = np.where(mask, DTYPE(0.0), ds)
    ds = ds * scale
    dq = ds @ k
    dk = np.ascontiguousarray(ds.transpose(0, 1, 3, 2)) @ q
    return dq, dk, dv


# ============================== Embedding ===============================
def embedding_forward(idx, W):
    return W[idx], (idx, W.shape)


def embedding_backward(dy, cache, vocab_size):
    idx, wshape = cache
    C = wshape[1]
    dW = np.zeros((vocab_size, C), dtype=DTYPE)
    flat = idx.reshape(-1)
    dy2 = dy.reshape(-1, C)
    np.add.at(dW, flat, dy2)
    return dW


# ============================== Residual ================================
def residual_forward(x, sub):
    return x + sub, (x, sub)


def residual_backward(dy, cache):
    """残差的两路梯度都为 dy 本身。"""
    return dy, dy


# ============================ Cross Entropy =============================
def cross_entropy_forward(logits, target):
    """logits:(N,V) target:(N,) int64 -> 标量 loss 与 cache"""
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    ez = np.exp(z)
    s = ez.sum(axis=-1, keepdims=True)
    logp = z - np.log(s)
    n = target.shape[0]
    loss = -logp[np.arange(n), target].mean()
    return DTYPE(loss), (logp, target, n)


def cross_entropy_backward(dout, cache):
    logp, target, n = cache
    d = np.exp(logp)
    d[np.arange(n), target] -= 1.0
    d *= DTYPE(dout / n)
    return d
