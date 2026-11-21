# -*- coding: utf-8 -*-
"""守护式续训：串行调用 train.py --resume，直到 latest.npz 的 step 达到 --target。

用途：CPU 长跑 + Windows 文件占用等偶发中断时自动接续，不需要人工盯守。
    python tools/keep_train.py --target 400 --batch 8 --log-every 10 --ckpt-every 100
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_step(ckpt_dir):
    """与 train.py 的 find_latest_ckpt 保持一致：latest.json 指针 -> 最大的 stepNNNNNN.npz。"""
    import json
    import numpy as np
    pj = os.path.join(ckpt_dir, "latest.json")
    if os.path.exists(pj):
        try:
            with open(pj, encoding="utf-8") as fh:
                info = json.load(fh)
            cand = os.path.join(ckpt_dir, info.get("file", ""))
            if os.path.exists(cand):
                with np.load(cand, allow_pickle=False) as z:
                    return int(z["step"])
        except Exception:
            pass
    best = 0
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.startswith("step") and f.endswith(".npz"):
                try:
                    with np.load(os.path.join(ckpt_dir, f), allow_pickle=False) as z:
                        best = max(best, int(z["step"]))
                except Exception:
                    pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--max-stall", type=int, default=5)
    args = ap.parse_args()

    ckpt_dir = os.path.join(ROOT, "ckpt")
    stall = 0
    while True:
        cur = read_step(ckpt_dir)
        if cur >= args.target:
            print("[keep] 已达成目标 step=%d，守护退出。" % cur, flush=True)
            return 0
        print("[keep] 当前 step=%d，续训至 %d（%s）"
              % (cur, args.target, time.strftime("%H:%M:%S")), flush=True)
        cmd = [sys.executable, "-X", "utf8", os.path.join(ROOT, "train.py"),
               "--steps", str(args.target - cur), "--total", str(args.target),
               "--batch", str(args.batch), "--lr", str(args.lr),
               "--warmup", str(args.warmup), "--log-every", str(args.log_every),
               "--ckpt-every", str(args.ckpt_every), "--resume"]
        rc = subprocess.call(cmd, cwd=ROOT)
        new = read_step(ckpt_dir)
        print("[keep] 子进程返回 %d，step=%d -> %d" % (rc, cur, new), flush=True)
        stall = 0 if new > cur else stall + 1
        if stall >= args.max_stall:
            print("[keep] 连续 %d 次无进展，停止自动续训，请人工查看 logs/train_run.log。" % stall, flush=True)
            return 1
        time.sleep(3)


if __name__ == "__main__":
    sys.exit(main())
