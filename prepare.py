# -*- coding: utf-8 -*-
"""语料预处理：训练手写 BPE → 把 corpus.txt 编码为定长 token 二进制。

产物：bp/tokenizer.json、data/train.bin（uint16 小端）、data/meta.json
用法：python prepare.py [--bpe-chars 8000000] [--vocab 16000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from bp.bpe import BPETokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe-chars", type=int, default=8_000_000)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "corpus.txt"))
    args = ap.parse_args()

    t0 = time.time()
    with open(args.corpus, encoding="utf-8", errors="replace") as f:
        text = f.read()
    log("语料读取完成：%.1f MB" % (len(text.encode("utf-8")) / 1e6))

    tk = BPETokenizer()
    log("开始训练 BPE（前 %d 字符，目标词表 %d）" % (args.bpe_chars, args.vocab))
    tk.train(text, vocab_size=args.vocab, max_chars=args.bpe_chars, verbose=True)
    tok_path = os.path.join(ROOT, "bp", "tokenizer.json")
    tk.save(tok_path)
    log("BPE 训练完成，merges=%d，耗时 %.1fs，已存 %s"
        % (len(tk.merges), time.time() - t0, tok_path))

    # 往返无损自检（中英文）
    probes = [
        "你好，我是人工智能助手。Hello, I'm an AI assistant.",
        "User: 请介绍一下量子力学。\nAssistant: 量子力学是描述微观粒子行为的物理理论。",
        "数字 0123456789 符号 !@#$%^&*() 混合 mixed 文本 test.",
    ]
    for s in probes:
        ids = tk.encode(s)
        back = tk.decode(ids)
        ok = back == s
        log("往返自检 %s：token=%d 还原一致=%s" % ("OK" if ok else "FAIL", len(ids), ok))
        if not ok:
            log("  原始: %r" % s)
            log("  还原: %r" % back)

    log("开始编码全量语料…")
    ids = tk.encode(text)
    log("编码完成：%d tokens（%.1f 字符/token）" % (len(ids), len(text) / max(1, len(ids))))
    arr = np.array(ids, dtype=np.uint16)
    bin_path = os.path.join(ROOT, "data", "train.bin")
    arr.tofile(bin_path)
    meta = {"vocab_size": args.vocab, "n_tokens": int(arr.size),
            "dtype": "uint16", "corpus_bytes": len(text.encode("utf-8")),
            "chars_per_token": round(len(text) / max(1, len(ids)), 3)}
    with open(os.path.join(ROOT, "data", "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log("已写出 %s（%.1f MB），总计耗时 %.1fs"
        % (bin_path, arr.nbytes / 1e6, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
