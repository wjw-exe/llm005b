# -*- coding: utf-8 -*-
"""对话语料下载与整理（纯标准库）。

- 只走 hf-mirror.com 镜像，全部请求带超时，失败自动切换备用数据集
- 统一整理成 User:/Assistant: 对话样本，输出 data/corpus.txt 与 data/pairs.jsonl
- 全过程写日志 logs/download.log，绝不长时间挂起
"""
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
os.makedirs(DATA, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
LOG_PATH = os.path.join(LOGS, "download.log")


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


MIRROR = "https://hf-mirror.com"

# 按顺序尝试：体积小、对话/指令格式明显
SOURCES = [
    ("dolly", "datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl", "jsonl"),
    ("alpaca", "datasets/tatsu-lab/alpaca/resolve/main/alpaca_data.json", "alpaca"),
    ("alpaca_zh", "datasets/shibing624/alpaca-zh/resolve/main/alpaca_data_zh_51k.json", "alpaca"),
    ("oasst1_zh", "datasets/shibing624/sharegpt_gpt4/resolve/main/sharegpt_gpt4.jsonl", "sharegpt"),
]


def fetch(url, dst, timeout=60, tries=3, max_bytes=None):
    """流式下载，单次连接超时 + 总时长上限，失败重试。"""
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.4.0"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as out:
                total = 0
                while True:
                    if time.time() - t0 > 600:
                        raise TimeoutError("单文件下载超过 600 秒，主动中止")
                    chunk = r.read(1 << 18)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
                    if max_bytes and total > max_bytes:
                        break
            size = os.path.getsize(dst)
            if size < 1024:
                raise IOError("文件过小 %d 字节" % size)
            log("下载成功 %s -> %.1f MB（第 %d 次尝试）" % (url.split("/resolve/")[-1], size / 1048576, attempt))
            return True
        except Exception as e:  # noqa: BLE001
            log("下载失败（第 %d/%d 次）：%s | %s: %s" % (attempt, tries, url, type(e).__name__, e))
            time.sleep(1.5 * attempt)
    return False


def norm_text(s, limit):
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s[:limit]


def load_pairs(kind, path):
    pairs = []
    if kind == "jsonl":
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                ctx = norm_text(o.get("context", ""), 200)
                instr = norm_text(o.get("instruction", ""), 300)
                if ctx:
                    instr = instr + "（背景：" + ctx + "）"
                resp = norm_text(o.get("response", ""), 700)
                if len(instr) > 1 and len(resp) > 1:
                    pairs.append((instr, resp))
    elif kind == "alpaca":
        with open(path, encoding="utf-8", errors="replace") as f:
            arr = json.load(f)
        for o in arr:
            instr = norm_text(o.get("instruction", ""), 300)
            extra = norm_text(o.get("input", ""), 200)
            if extra:
                instr = instr + "（补充：" + extra + "）"
            resp = norm_text(o.get("output", ""), 700)
            if len(instr) > 1 and len(resp) > 1:
                pairs.append((instr, resp))
    elif kind == "sharegpt":
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                conv = o.get("conversations") or o.get("conversation") or []
                for i in range(len(conv) - 1):
                    a, b = conv[i], conv[i + 1]
                    if not isinstance(a, dict) or not isinstance(b, dict):
                        continue
                    if a.get("from") in ("human", "user") and b.get("from") in ("gpt", "assistant"):
                        p = norm_text(a.get("value", ""), 300)
                        r = norm_text(b.get("value", ""), 700)
                        if len(p) > 1 and len(r) > 1:
                            pairs.append((p, r))
    return pairs


def main():
    log("=== 语料准备开始（镜像 %s）===" % MIRROR)
    all_pairs = []
    used = []
    for name, rel, kind in SOURCES:
        if len(all_pairs) >= 40000:
            break
        dst = os.path.join(DATA, "src_%s" % name)
        url = MIRROR + "/" + rel
        log("尝试数据集 %s" % name)
        if not fetch(url, dst):
            log("跳过数据集 %s（下载失败）" % name)
            continue
        try:
            pairs = load_pairs(kind, dst)
        except Exception as e:  # noqa: BLE001
            log("解析失败 %s：%s" % (name, e))
            continue
        log("解析 %s 得到 %d 条对话样本" % (name, len(pairs)))
        if pairs:
            all_pairs.extend(pairs)
            used.append(name)
        if len(all_pairs) >= 40000:
            break

    if not all_pairs:
        log("严重错误：所有镜像数据集均不可用，语料为空")
        sys.exit(2)

    seen = set()
    uniq = []
    for p, r in all_pairs:
        key = p[:120]
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, r))

    txt_parts = []
    with open(os.path.join(DATA, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for p, r in uniq:
            f.write(json.dumps({"user": p, "assistant": r}, ensure_ascii=False) + "\n")
            txt_parts.append("User: " + p + "\nAssistant: " + r + "\n\n")

    corpus = os.path.join(DATA, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        f.write("".join(txt_parts))

    size = os.path.getsize(corpus) / 1048576
    log("=== 完成：样本 %d 条（去重后），语料 %.2f MB，使用数据集 %s ===" % (len(uniq), size, ",".join(used)))
    log("输出：%s" % corpus)


if __name__ == "__main__":
    main()
