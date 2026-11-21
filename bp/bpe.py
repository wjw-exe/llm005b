# -*- coding: utf-8 -*-
"""手写字节级 BPE 分词器（不依赖 tokenizers / sentencepiece）。

* 训练：从纯文本统计词频，逐步合并最高频相邻符号对，产出 merges 列表
* 编码：按 GPT-2 字节级 unicode 映射，任何 UTF-8 文本（含中文）都能无损编码
* 解码：把 token id 还原为字节串再按 utf-8 解码

token id 布局：0=<pad> 1=<eos> 2=<unk>，3..258 为 256 个字节，259+ 为合并符号
"""
import json
import re
import sys
from collections import Counter, defaultdict

PAD, EOS, UNK = 0, 1, 2
BYTE_OFFSET = 3
N_BYTES = 256
MERGES_START = BYTE_OFFSET + N_BYTES  # 259

PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[A-Za-z]+"
    r"| ?[0-9]+"
    r"| ?[\u4e00-\u9fff]"
    r"| ?[^\sA-Za-z0-9\u4e00-\u9fff]+"
    r"|\s+(?!\S)|\s+"
)


def _to_symbols(chunk):
    return list(chunk.encode("utf-8"))


class BPETokenizer:
    def __init__(self, merges=None, vocab_size=16000):
        self.merges = [tuple(m) for m in (merges or [])]
        self.vocab_size = vocab_size
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache = {}
        self._build_bytes()

    def _build_bytes(self):
        """预展开每个合并符号对应的原始字节，decode 时 O(1) 取用。"""
        self.merged = []
        for a, b in self.merges:
            self.merged.append(self._tok_bytes(a) + self._tok_bytes(b))

    def _tok_bytes(self, i):
        return bytes([i]) if i < N_BYTES else self.merged[i - MERGES_START]

    # ---------------------------------------------------------------- 训练
    @staticmethod
    def _word_freqs(text, max_chars=None):
        if max_chars:
            text = text[:max_chars]
        freq = Counter()
        for chunk in PATTERN.findall(text):
            if not chunk:
                continue
            freq[tuple(_to_symbols(chunk))] += 1
        return freq

    def train(self, text, vocab_size=16000, max_chars=None, verbose=False):
        freq = self._word_freqs(text, max_chars)
        words = {w: c for w, c in freq.items() if len(w) > 1}
        singles = {w: c for w, c in freq.items() if len(w) == 1}

        pair_counts = Counter()
        pair_where = defaultdict(set)
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pair_counts[(a, b)] += c
                pair_where[(a, b)].add(w)
        for w, c in singles.items():
            words[w] = c

        target = vocab_size - MERGES_START
        merges = []
        import heapq
        heap = [(-c, a, b) for (a, b), c in pair_counts.items()]
        heapq.heapify(heap)

        while len(merges) < target and heap:
            negc, a, b = heapq.heappop(heap)
            cur = pair_counts.get((a, b), 0)
            if cur <= 1 or -negc != cur:
                if cur > 1 and -negc != cur:
                    heapq.heappush(heap, (-cur, a, b))
                continue
            new_id = MERGES_START + len(merges)
            merges.append((a, b))
            affected = list(pair_where.pop((a, b), ()))
            for w in affected:
                cnt = words.pop(w, 0)
                if not cnt:
                    continue
                # 移除旧贡献
                for x, y in zip(w, w[1:]):
                    pair_counts[(x, y)] -= cnt
                    if pair_counts[(x, y)] <= 0:
                        pair_counts.pop((x, y), None)
                    s = pair_where.get((x, y))
                    if s is not None:
                        s.discard(w)
                # 合并
                nw = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        nw.append(new_id)
                        i += 2
                    else:
                        nw.append(w[i])
                        i += 1
                nw = tuple(nw)
                # 加入新贡献
                for x, y in zip(nw, nw[1:]):
                    pair_counts[(x, y)] += cnt
                    pair_where[(x, y)].add(nw)
                heapq.heappush(heap, (-cnt, new_id, -1))  # 占位，保证堆非空
                if len(nw) > 1:
                    for (x, y), c2 in (((x, y), pair_counts[(x, y)]) for x, y in zip(nw, nw[1:])):
                        heapq.heappush(heap, (-c2, x, y))
                words[nw] = words.get(nw, 0) + cnt
            if verbose and len(merges) % 1000 == 0:
                print("  merge %d/%d, 唯一词 %d" % (len(merges), target, len(words)), flush=True)

        self.merges = merges
        self.vocab_size = vocab_size
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self._cache = {}
        self._build_bytes()
        return self

    # ---------------------------------------------------------------- 编码
    def _bpe(self, symbols):
        w = tuple(symbols)
        if len(w) < 2:
            return list(w)
        cached = self._cache.get(w)
        if cached is not None:
            return cached
        ranks = self.ranks
        while len(w) > 1:
            best = None
            best_rank = None
            for i in range(len(w) - 1):
                r = ranks.get((w[i], w[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = (w[i], w[i + 1])
            if best is None:
                break
            a, b = best
            nw = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(MERGES_START + best_rank)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            w = tuple(nw)
        out = list(w)
        if len(self._cache) < 200000:
            self._cache[tuple(symbols)] = out
        return out

    def encode(self, text, add_eos=False):
        ids = []
        for chunk in PATTERN.findall(text):
            if not chunk:
                continue
            ids.extend(self._bpe(_to_symbols(chunk)))
        # 字节符号 0..255 → 3..258；合并符号 id 已是 259+，保持不变
        ids = [x + BYTE_OFFSET if x < N_BYTES else x for x in ids]
        if add_eos:
            ids.append(EOS)
        return ids

    def token_bytes(self, i):
        """返回单个 token 对应的原始字节（用于流式解码）。"""
        if i < BYTE_OFFSET:
            return b""
        if i < MERGES_START:
            return bytes([i - BYTE_OFFSET])
        j = i - MERGES_START
        return self.merged[j] if 0 <= j < len(self.merged) else b""

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if i < BYTE_OFFSET:
                continue
            if i < MERGES_START:
                out.append(i - BYTE_OFFSET)
            else:
                j = i - MERGES_START
                if 0 <= j < len(self.merged):
                    out.extend(self.merged[j])
        return out.decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- 存取
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "vocab_size": self.vocab_size,
                       "merges": [[a, b] for a, b in self.merges]}, f)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tk = BPETokenizer(d["merges"], d.get("vocab_size", MERGES_START + len(d["merges"])))
        return tk


if __name__ == "__main__":
    # 自检：中英文往返无损
    sample = "你好，世界！Hello world, 这是一个 BPE 分词器自检。数字 12345 与符号 @#$ 也应无损。"
    tk = BPETokenizer()
    tk.train(sample * 300, vocab_size=1000)
    ids = tk.encode(sample)
    back = tk.decode(ids)
    print("原始:", sample)
    print("还原:", back)
    print("往返一致:", back == sample, "token 数:", len(ids))
    sys.exit(0 if back == sample else 1)
