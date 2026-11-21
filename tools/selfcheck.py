# -*- coding: utf-8 -*-
"""项目自检：确认零深度学习框架依赖 + 参数量明细 + 分词器往返 + 权重完整性。

用法：python tools/selfcheck.py
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BANNED = ("torch", "transformers", "jax", "onnx", "tensorflow", "keras", "paddle",
          "mxnet", "sklearn", "scipy", "einops", "safetensors")

# 第三方 / 禁用库扫描（只看 import 行）
def scan_imports():
    pat = re.compile(r"^\s*(?:from|import)\s+([A-Za-z0-9_\.]+)")
    found = {}
    files = []
    for d in ("", "bp", "tools"):
        files += glob.glob(os.path.join(ROOT, d, "*.py"))
    for f in sorted(files):
        rel = os.path.relpath(f, ROOT)
        for line in open(f, encoding="utf-8", errors="replace"):
            m = pat.match(line)
            if not m:
                continue
            mod = m.group(1).split(".")[0]
            found.setdefault(mod, set()).add(rel)
    return found, [os.path.relpath(f, ROOT) for f in sorted(files)]


def main():
    print("=" * 70)
    print("llm005b 自检报告")
    print("=" * 70)

    found, files = scan_imports()
    print("\n[1] 依赖审计（%d 个源文件）" % len(files))
    for f in files:
        print("    - %s" % f)
    banned_hit = [m for m in BANNED if m in found]
    print("\n    引用的顶层模块：%s" % ", ".join(sorted(found)))
    if banned_hit:
        print("    [FAIL] 发现被禁止的深度学习/科学计算框架：%s" % ", ".join(banned_hit))
    else:
        print("    [PASS] 未引用任何深度学习框架（torch/transformers/jax/onnx/tensorflow 等）")
    print("    [INFO] 第三方依赖仅限：%s" % ", ".join(
        m for m in sorted(found) if m in ("numpy",)))

    print("\n[2] 参数量明细")
    from model import GPT, default_config
    cfg = default_config()
    cp = os.path.join(ROOT, "configs", "base.json")
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in cfg})
    C, L, V, F, T = cfg["d_model"], cfg["n_layer"], cfg["vocab_size"], cfg["ffn_dim"], cfg["ctx"]
    n_wte = V * C
    n_wpe = T * C
    n_per_layer = (2 * C) + (4 * C * C) + (4 * C) + (2 * C) + (C * F + F) + (F * C + C)
    n_final = 2 * C
    total = n_wte + n_wpe + L * n_per_layer + n_final
    print("    wte   (vocab {} x {})        : {:>12,}".format(V, C, n_wte))
    print("    wpe   (ctx {} x {})          : {:>12,}".format(T, C, n_wpe))
    print("    per-layer (ln+attn+ffn)      : {:>12,}   x {} 层".format(n_per_layer, L))
    print("    final layernorm              : {:>12,}".format(n_final))
    print("    {:<28} : {:>12,}  ({:.2f}M)".format("合计", total, total / 1e6))
    m = GPT(cfg)
    real = m.num_params()
    print("    {:<28} : {:>12,}  ({:.2f}M)  {}".format(
        "实测 GPT.num_params()", real, real / 1e6,
        "[PASS 一致]" if real == total else "[FAIL 不一致]"))

    print("    是否落在 45M-55M 区间          : %s" % ("[PASS]" if 45e6 <= real <= 55e6 else "[FAIL]"))

    print("\n[3] 分词器往返自检")
    from bp.bpe import BPETokenizer
    tk_path = os.path.join(ROOT, "bp", "tokenizer.json")
    if not os.path.exists(tk_path):
        print("    [SKIP] 尚未训练分词器（bp/tokenizer.json 不存在）")
    else:
        tk = BPETokenizer.load(tk_path)
        probes = ["你好，我是小模型。", "User: 你是谁？\nAssistant: 我是一个参数量很小的语言模型。",
                  "混合 mixed 中英 text 123 !@#"]
        ok_all = True
        for s in probes:
            ids = tk.encode(s)
            ok = tk.decode(ids) == s
            ok_all = ok_all and ok
            print("    %s  %2d token  %r" % ("[OK]" if ok else "[FAIL]", len(ids), s[:36]))
        print("    词表大小：merges=%d（含 256 字节符号与特殊位）" % len(tk.merges))

    print("\n[4] 权重完整性")
    ck = None
    for name in ("final.npz", "latest.npz"):
        p = os.path.join(ROOT, "ckpt", name)
        if os.path.exists(p):
            ck = p
            break
    if ck is None:
        print("    [SKIP] ckpt/ 下暂无权重文件")
    else:
        import numpy as np
        z = np.load(ck, allow_pickle=False)
        miss = [k for k in m.params if k not in z.files]
        print("    文件：%s（%.1f MB）" % (os.path.relpath(ck, ROOT), os.path.getsize(ck) / 1e6))
        print("    步数：%s  loss：%.4f" % (int(z["step"]) if "step" in z.files else -1,
                                          float(z["loss"]) if "loss" in z.files else float("nan")))
        print("    缺失参数：%s" % ("无 [PASS]" if not miss else miss))
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
