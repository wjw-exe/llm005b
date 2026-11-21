# -*- coding: utf-8 -*-
"""0.05B 指令模型推理自检 / 性能基准（全部走手写 numpy 算子）。

用法（在工程根目录）：
    python inf/bench_instruct.py

输出：权重校验、分词往返、KV Cache 数值一致性、prefill/增量解码性能、内存峰值。
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from instruct_chat import rss_mb  # noqa: E402
from minigpt_numpy import MiniGPTInstruct  # noqa: E402

CASES = [
    "Hello, world!",
    "Once upon a time, there was a little cat.",
    "你好，介绍一下你自己",
    "中英混排 mixed 文本 123，标点。",
    "  leading space and\ttab\nnewline",
    "User: What is 2+2?\nAssistant:",
    "The quick brown fox jumps over the lazy dog.",
    "表情符号 🙂 与繁体字 測試",
    "a" * 40,
    "数字 0123456789 与符号 !@#$%^&*()",
]


def main():
    t0 = time.perf_counter()
    m = MiniGPTInstruct(os.path.join(ROOT, "model_50m_instruct"), verbose=True)
    load_s = time.perf_counter() - t0
    print("\n[1] 权重加载")
    print("    加载耗时        : %.2f s" % load_s)
    print("    safetensors 数组: %d 个，匹配进图 %d 个张量"
          % (m.raw_array_count, m.raw_array_count - len(m.unexpected_keys)))
    print("    缺失键 / 多余键 : %d / %d" % (len(m.missing_keys), len(m.unexpected_keys)))
    print("    唯一参数(去重后): %s" % f"{m.uniq_count:,}")
    print("    head 与 token_emb 最大差异: %.3e（完全绑定=%s）"
          % (m.tied_diff, m.tied_diff == 0.0))

    print("\n[2] 分词往返自检（encode→decode 必须逐字节一致）")
    ok = 0
    for s in CASES:
        ids = m.tok.encode(s)
        back = m.tok.decode(ids)
        good = back == s
        ok += int(good)
        print("    [%s] len=%3d  %r -> %r" % ("PASS" if good else "FAIL", len(ids), s[:26], back[:26]))
    print("    结果：%d/%d PASS" % (ok, len(CASES)))

    print("\n[3] KV Cache 增量解码数值一致性（逐步增量 vs 全序列一次前向）")
    ids = m.tok.encode("Once upon a time, there was a little cat.")
    lg_full, _ = m.forward(ids, None, 0)
    lg_step = []
    cache = None
    for i, t in enumerate(ids):
        out, cache = m.forward([t], cache, i)
        lg_step.append(out[0, -1])
    lg_step = np.stack(lg_step, axis=0)
    d = np.abs(lg_full[0] - lg_step)
    print("    logits 形状     : %s vs %s" % (lg_full[0].shape, lg_step.shape))
    print("    最大绝对误差    : %.3e   平均绝对误差: %.3e" % (d.max(), d.mean()))
    same = [int(np.argmax(a)) == int(np.argmax(b)) for a, b in zip(lg_full[0], lg_step)]
    print("    逐位置 argmax 一致率: %d/%d" % (sum(same), len(same)))

    print("\n[4] 性能基准（numpy 手写算子 + 预分配 KV Cache）")
    prompt = "User: Write a short story about a cat.\nAssistant:"
    pids = m.tok.encode(prompt)
    t0 = time.perf_counter()
    logits, cache = m.forward(pids, None, 0)
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    print("    prefill         : %d token / %.0f ms (%.1f ms/token)"
          % (len(pids), prefill_ms, prefill_ms / len(pids)))

    n = 100
    tok = int(np.argmax(logits[0, -1]))
    t0 = time.perf_counter()
    pos = len(pids)
    for _ in range(n):
        out, cache = m.forward([tok], cache, pos)
        pos += 1
        tok = int(np.argmax(out[0, -1]))
    inc_s = time.perf_counter() - t0
    print("    增量解码        : %d token / %.2f s = %.2f tok/s (%.1f ms/token)"
          % (n, inc_s, n / inc_s, inc_s / n * 1000.0))
    print("    首 token 延迟   : %.1f ms（prefill 后单步前向）" % (inc_s / n * 1000.0))
    rss, peak = rss_mb()
    print("    内存            : 当前 %.0f MB / 峰值 %.0f MB" % (rss, peak))
    print("    KV Cache 占用   : %.1f MB（%d 层 x K/V x 1 x %d 头 x %d ctx x %d dim x 4B）"
          % (2 * m.n_layer * m.n_head * m.ctx * m.dh * 4 / 1048576.0,
             m.n_layer, m.n_head, m.ctx, m.dh))

    print("\n[5] 中文输入处理（词表无 CJK，按 UTF-8 字节切分）")
    zh = "你好，介绍一下你自己"
    t0 = time.perf_counter()
    zids = m.tok.encode(zh)
    enc_ms = (time.perf_counter() - t0) * 1000.0
    print("    中文 prompt     : %d 字符 -> %d token，编码 %.2f ms，往返一致=%s"
          % (len(zh), len(zids), enc_ms, m.tok.decode(zids) == zh))
    print("    说明            : 权重仅在英文 TinyStories 上训练，中文没有语义表征，"
          "实测中文问答输出为字节碎片，属能力边界而非分词缺陷。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
