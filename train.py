# -*- coding: utf-8 -*-
"""从零训练器：手写 AdamW + warmup/cosine + 梯度裁剪 + 断点续训 + 分步日志。

只依赖 numpy。示例：
  python train.py --steps 1200                    # 前台训练
  python train.py --steps 1200 --background       # 后台训练（立即返回，日志轮询）
  python train.py --resume                        # 从 checkpoints/latest.npz 续训
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from model import GPT, default_config  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------- AdamW
class AdamW:
    """手写 AdamW：解耦权重衰减，动量二阶矩偏差修正。"""

    def __init__(self, params, lr=6e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, grad_clip=1.0):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.clip = grad_clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self._bufs = {}
        self._bufs2 = {}

    @staticmethod
    def _no_decay(name):
        # 偏置、LayerNorm 的缩放/平移不施加权重衰减
        return (name.endswith("_g") or name.endswith("_b") or name.endswith("b1")
                or name.endswith("b2") or name.endswith("qb") or name.endswith("kb")
                or name.endswith("vb") or name.endswith("ob"))

    def clip_grads(self, grads):
        total = 0.0
        for g in grads.values():
            total += float(np.sum(g * g))
        norm = math.sqrt(total)
        scale = 1.0
        if self.clip > 0 and norm > self.clip:
            scale = self.clip / (norm + 1e-12)
            for k in grads:
                grads[k] *= scale
        return norm, scale

    def step(self, params, grads, t, lr):
        """原地更新：临时数组全部复用，避免每步数百 MB 的分配/释放开销。"""
        bc1 = 1.0 - self.b1 ** t
        bc2 = 1.0 - self.b2 ** t
        lr_t = lr / bc1
        inv_bc2 = 1.0 / bc2
        for k, p in params.items():
            g = grads[k]
            m = self.m[k]
            v = self.v[k]
            d = self._buf(k, g)
            u = self._buf2(k, g)
            # m = b1*m + (1-b1)*g
            m *= self.b1
            np.multiply(g, 1.0 - self.b1, out=u)
            m += u
            # v = b2*v + (1-b2)*g^2（g 就地平方，此后不再需要原值）
            np.multiply(g, g, out=g)
            v *= self.b2
            g *= (1.0 - self.b2)
            v += g
            # denom = sqrt(v/bc2) + eps
            np.multiply(v, inv_bc2, out=d)
            np.sqrt(d, out=d)
            d += self.eps
            # upd = lr_t * m / denom
            np.divide(m, d, out=u)
            u *= lr_t
            if not self._no_decay(k):
                p *= (1.0 - lr * self.wd)
            p -= u

    def _buf(self, k, ref):
        b = self._bufs.get(k)
        if b is None or b.shape != ref.shape or b.dtype != ref.dtype:
            b = np.empty_like(ref)
            self._bufs[k] = b
        return b

    def _buf2(self, k, ref):
        b = self._bufs2.get(k)
        if b is None or b.shape != ref.shape or b.dtype != ref.dtype:
            b = np.empty_like(ref)
            self._bufs2[k] = b
        return b

    def state(self):
        return {"adam_m:" + k: v for k, v in self.m.items()} | \
               {"adam_v:" + k: v for k, v in self.v.items()}

    def load_state(self, arrays):
        for k in self.m:
            mk = "adam_m:" + k
            vk = "adam_v:" + k
            if mk in arrays:
                self.m[k] = arrays[mk].astype(np.float32, copy=False)
                self.v[k] = arrays[vk].astype(np.float32, copy=False)


def save_pointer(ckpt_dir, fname, step, tokens, best, loss):
    """写一个小体积指针文件（<200B），大权重文件永不覆盖写，规避 Windows 文件占用冲突。"""
    info = {"file": fname, "step": int(step), "tokens": int(tokens),
            "best": float(best), "loss": float(loss),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(ckpt_dir, "latest.json")
    tmp = path + ".writing"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    for _ in range(10):
        try:
            os.replace(tmp, path)
            return info
        except OSError:
            time.sleep(0.5)
    return info


def _step_of(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            return int(z["step"]), int(z["tokens"]), float(z["loss"]) if "loss" in z else float("nan")
    except Exception:
        return -1, 0, float("nan")


def find_latest_ckpt(ckpt_dir):
    """优先级：latest.json 指针 -> final.npz -> 步号最大的 stepNNNNNN.npz（兼容旧 latest.npz）。"""
    pj = os.path.join(ckpt_dir, "latest.json")
    if os.path.exists(pj):
        try:
            with open(pj, encoding="utf-8") as fh:
                info = json.load(fh)
            cand = os.path.join(ckpt_dir, info.get("file", ""))
            if os.path.exists(cand) and _step_of(cand)[0] >= 0:
                return cand
        except Exception:
            pass
    fin = os.path.join(ckpt_dir, "final.npz")
    if os.path.exists(fin) and _step_of(fin)[0] >= 0:
        return fin
    cands = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("step") and f.endswith(".npz"))
    for f in reversed(cands):
        full = os.path.join(ckpt_dir, f)
        if _step_of(full)[0] >= 0:
            return full
    old = os.path.join(ckpt_dir, "latest.npz")
    if os.path.exists(old) and _step_of(old)[0] >= 0:
        return old
    return None


# ------------------------------------------------------------- 学习率
def lr_at(step, total, warmup, base_lr, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(1.0, max(0.0, p))
    cos = 0.5 * (1.0 + math.cos(math.pi * p))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cos)


def get_batch(ids, B, T, rng):
    n = len(ids)
    ix = rng.integers(0, max(1, n - T - 2), size=B)
    x = np.stack([ids[int(i):int(i) + T] for i in ix]).astype(np.int64)
    y = np.stack([ids[int(i) + 1:int(i) + 1 + T] for i in ix]).astype(np.int64)
    return x, y


def save_ckpt(path, model, opt, step, cfg, tokens, best, loss=float("nan")):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    arrs = {"config": np.array(json.dumps(cfg, ensure_ascii=False)),
            "step": np.array(step), "loss": np.array(float(loss), dtype=np.float64),
            "tokens": np.array(tokens), "best": np.array(best, dtype=np.float64)}
    arrs.update(model.state_arrays())
    arrs.update(opt.state())
    tmp = path + ".writing"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrs)
    err = None
    for _ in range(8):
        try:
            os.replace(tmp, path)
            err = None
            break
        except OSError as e:      # Windows 下目标可能被杀软/外部读取短暂占用
            err = e
            time.sleep(1.5)
    if err is not None:
        raise err
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--total", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--data", default=os.path.join("data", "train.bin"), help="语料 bin 路径")
    ap.add_argument("--ckpt-dir", default="ckpt", help="checkpoint 目录")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-file", default=os.path.join(ROOT, "logs", "train.log"))
    ap.add_argument("--max-hours", type=float, default=6.0)
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)

    # ---------------- 后台模式：脱离控制台重新启动自身后立即返回
    if args.background:
        logf = os.path.join(ROOT, "logs", "train_run.log")
        cmd = [sys.executable, os.path.abspath(__file__)]
        for k, v in vars(args).items():
            if k in ("background", "log_file"):
                continue
            if isinstance(v, bool):
                if v:
                    cmd.append("--" + k.replace("_", "-"))
            else:
                cmd += ["--" + k.replace("_", "-"), str(v)]
        fh = open(logf, "ab")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, close_fds=True,
                                creationflags=0x00000008 | 0x00000200)
        print("已在后台启动训练，PID=%d" % proc.pid)
        print("实时日志：%s" % args.log_file)
        print("轮询命令：python -c \"import tailer\" 或 type logs\\train.log")
        return 0

    cfg = default_config()
    cfg_path = os.path.join(ROOT, "configs", "base.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    ids = np.memmap(os.path.join(ROOT, args.data), dtype=np.uint16, mode="r")
    meta_path = os.path.join(ROOT, "data", "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        cfg["vocab_size"] = int(meta["vocab_size"])
    print("语料 tokens 总量：%d（%.1f MB uint16）" % (len(ids), ids.nbytes / 1e6), flush=True)

    model = GPT(cfg, seed=args.seed)
    n_par = model.num_params()
    print("模型参数量：%d（%.2fM）｜d_model=%d n_layer=%d n_head=%d ctx=%d ffn=%d tied=%s"
          % (n_par, n_par / 1e6, cfg["d_model"], cfg["n_layer"], cfg["n_head"],
             cfg["ctx"], cfg["ffn_dim"], cfg["tied"]), flush=True)

    opt = AdamW(model.params, lr=args.lr, weight_decay=args.wd, grad_clip=args.clip)
    start_step, tokens_seen, best = 0, 0, 1e9
    ckpt_dir = os.path.join(ROOT, args.ckpt_dir)
    latest = find_latest_ckpt(ckpt_dir) if args.resume else None
    if latest:
        with np.load(latest, allow_pickle=False) as z:
            model.load_state(z)
            opt.load_state(z)
            start_step = int(z["step"])
            tokens_seen = int(z["tokens"])
            best = float(z["best"]) if "best" in z else 1e9
        print("已从 %s 恢复：step=%d tokens=%d" % (latest, start_step, tokens_seen), flush=True)
    elif args.resume:
        print("未找到可用 checkpoint，将从头开始训练。", flush=True)

    rng = np.random.default_rng(args.seed + start_step)
    lf = open(args.log_file, "a", encoding="utf-8")
    hdr = ("# 训练日志 | config: d_model=%d n_layer=%d n_head=%d ctx=%d ffn=%d vocab=%d "
           "params=%d | batch=%d lr=%g warmup=%d wd=%g clip=%g"
           % (cfg["d_model"], cfg["n_layer"], cfg["n_head"], cfg["ctx"], cfg["ffn_dim"],
              cfg["vocab_size"], n_par, args.batch, args.lr, args.warmup, args.wd, args.clip))
    lf.write(hdr + "\n")
    lf.write("# step | loss | lr | tokens/s | 累计tokens | 步耗时s | 梯度范数\n")
    lf.flush()
    print(hdr, flush=True)

    t_start = time.time()
    running_loss = None
    total = start_step + args.steps
    plan_total = args.total if args.total > 0 else total
    for step in range(start_step, total):
        tl = time.time()
        lr = lr_at(step, plan_total, args.warmup, args.lr)
        x, y = get_batch(ids, args.batch, cfg["ctx"], rng)
        model.zero_grads()
        loss = model.forward(x, y)
        model.backward()
        gnorm, gscale = opt.clip_grads(model.grads)
        opt.step(model.params, model.grads, step + 1, lr)
        dt = time.time() - tl
        tokens_seen += args.batch * cfg["ctx"]
        lv = float(loss)
        running_loss = lv if running_loss is None else 0.98 * running_loss + 0.02 * lv
        if (step + 1) % args.log_every == 0 or step == start_step:
            el = time.time() - t_start
            line = ("%6d | %.4f | %.3e | %6.0f | %9d | %5.2f | %.3f%s"
                    % (step + 1, lv, lr, (step + 1 - start_step) * args.batch * cfg["ctx"] / max(el, 1e-9),
                       tokens_seen, dt, gnorm, "" if gscale == 1.0 else " (clipped)"))
            lf.write(line + "\n")
            lf.flush()
            print("step " + line, flush=True)
        if (step + 1) % args.ckpt_every == 0:
            best = min(best, lv)
            # 先落盘带步号的历史权重，再更新 latest 指针，避免一次写失败丢掉全部进度
            name = "step%06d.npz" % (step + 1)
            sz = save_ckpt(os.path.join(ckpt_dir, name), model, opt, step + 1,
                           cfg, tokens_seen, best, lv)
            save_pointer(ckpt_dir, name, step + 1, tokens_seen, best, lv)
            print("  [ckpt] step=%d 已保存（%.0f MB）" % (step + 1, sz / 1e6), flush=True)
        if time.time() - t_start > args.max_hours * 3600:
            print("达到 --max-hours 上限，提前停止并保存。", flush=True)
            break

    fin_step = min(total, step + 1)
    save_ckpt(os.path.join(ckpt_dir, "final.npz"), model, opt, fin_step,
              cfg, tokens_seen, min(best, lv), lv)
    save_pointer(ckpt_dir, "final.npz", fin_step, tokens_seen, min(best, lv), lv)
    lf.close()
    print("训练结束：step=%d tokens=%d 最终 loss=%.4f 用时 %.1f 分钟"
          % (step + 1, tokens_seen, lv, (time.time() - t_start) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
