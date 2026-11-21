# -*- coding: utf-8 -*-
"""实测脚本：运行 chat_demo 并采样子进程内存峰值（Windows psapi），输出真实性能证据。

用法：python tools/mem_probe.py [--ckpt ckpt/final.npz] [--max-new 64]
产物：logs/mem_probe.json
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def sample(pid):
    psapi = ctypes.WinDLL("psapi")
    kernel32 = ctypes.WinDLL("kernel32")
    h = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not h:
        return 0
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ok = psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    kernel32.CloseHandle(h)
    return pmc.PeakWorkingSetSize if ok else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("ckpt", "final.npz"))
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    cmd = [sys.executable, "-X", "utf8", os.path.join("tools", "chat_demo.py"),
           "--ckpt", args.ckpt, "--max-new", str(args.max_new),
           "--out", os.path.join("logs", "chat_mem.txt")]
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    peak = 0
    while p.poll() is None:
        peak = max(peak, sample(p.pid))
        time.sleep(0.15)
    out = p.stdout.read() or ""
    wall = time.time() - t0

    lines = [l for l in out.splitlines() if "tok/s" in l]
    toks = []
    for l in lines:
        try:
            toks.append(float(l.split("=")[1].split("tok/s")[0].strip()))
        except Exception:
            pass
    res = {
        "ckpt": args.ckpt,
        "wall_s": round(wall, 2),
        "peak_workset_mb": round(peak / 1048576, 1),
        "tok_per_s_per_round": toks,
        "avg_tok_per_s": round(sum(toks) / len(toks), 2) if toks else None,
        "generated_text": out[-1200:],
    }
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "mem_probe.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "generated_text"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
