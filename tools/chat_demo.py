# -*- coding: utf-8 -*-
"""非交互演示脚本：把若干问题按顺序喂给 chat.py，并把完整会话记录到文件。

用途：在没有人工输入的环境里复现"多轮真实对话"证据。
用法：
    python tools/chat_demo.py                        # 默认 3 轮，写入 logs/chat_demo.txt
    python tools/chat_demo.py --ckpt ckpt/final.npz --out logs/chat_demo.txt
    python tools/chat_demo.py --questions "你好" "太阳为什么发光"
"""
import argparse
import io
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_QUESTIONS = [
    "你好，请简单介绍一下你自己。",
    "那你能帮我做什么？",
    "谢谢，再见。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "chat_demo.txt"))
    ap.add_argument("--questions", nargs="*", default=None)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    qs = args.questions or DEFAULT_QUESTIONS
    cmd = [sys.executable, "-X", "utf8", os.path.join(ROOT, "chat.py"),
           "--max-new", str(args.max_new), "--temp", str(args.temp)]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]

    payload = "\n".join(qs + ["/exit"]) + "\n"
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, input=payload, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=args.timeout)
    wall = time.time() - t0
    text = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else "")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("chat.py 真实对话记录（自动喂入 %d 轮，非人工编辑）\n" % len(qs))
        f.write("时间：%s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        f.write("命令：%s\n" % " ".join(cmd))
        f.write("墙钟耗时：%.1f 秒   返回码：%d\n" % (wall, proc.returncode))
        f.write("=" * 72 + "\n\n")
        f.write(text)
    print(text)
    print("\n[已写入] %s（%.1f 秒）" % (args.out, wall))
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
