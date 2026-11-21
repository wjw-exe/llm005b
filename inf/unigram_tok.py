# -*- coding: utf-8 -*-
"""Unigram 分词器手写实现（对齐 hayder86al/tinystories-50m-instruct）。

该仓库 tokenizer.json 的实际情况（逐字段实测）：
  model.type      = "Unigram"      （不是 BPE）
  model.vocab     = list[[piece, log_score], ...]  8000 项
  byte_fallback   = False
  unk_id = 2, 特殊 token: <pad>=0, <eos>=1, <unk>=2
  normalizer      = None
  pre_tokenizer   = ByteLevel(add_prefix_space=True, use_regex=True)
  decoder         = ByteLevel
  post_processor  = TemplateProcessing（无额外特殊 token，对话靠纯文本模板）

因此管线为：GPT-2 风格正则预切分 -> 字节级 unicode 映射 -> Viterbi 最大对数似然切分
-> 反向映射还原。词表只覆盖 256 个字节映射字符（无任何 CJK token），
所以中文以 UTF-8 字节级 token 表示（1 汉字 = 3 token），可无损往返但效率低。
"""
import json
import re
from typing import Dict, List, Tuple

# ---------------------- GPT-2 字节 <-> unicode 映射 ----------------------
def _bytes_to_unicode() -> Dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = _bytes_to_unicode()
U2B = {u: b for b, u in B2U.items()}

# GPT-2 预切分正则（\p{L} 用 [^\W\d_] 近似，CJK 亦归入字母类）
_PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")


def _split_pieces(text: str) -> List[str]:
    """GPT-2 风格预切分；保证覆盖全部字符（gap 单独成段，绝不丢字）。"""
    out: List[str] = []
    pos = 0
    for m in _PAT.finditer(text):
        if m.start() > pos:
            out.append(text[pos:m.start()])
        if m.group():
            out.append(m.group())
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return [p for p in out if p]


class UnigramTokenizer:
    def __init__(self, tokenizer_json: str):
        with open(tokenizer_json, "r", encoding="utf-8") as f:
            tk = json.load(f)
        model = tk["model"]
        if model.get("type") != "Unigram":
            raise ValueError(f"非 Unigram 模型: {model.get('type')}")
        self.vocab: Dict[str, int] = {}
        self.scores: List[float] = []
        for idx, (piece, score) in enumerate(model["vocab"]):
            self.vocab[piece] = idx
            self.scores.append(float(score))
        self.unk_id = int(model.get("unk_id", 2))
        self.byte_fallback = bool(model.get("byte_fallback", False))
        self.unk_score = (min(self.scores) - 10.0) if self.scores else -20.0
        self.maxlen = max((len(p) for p in self.vocab), default=1)
        self.special = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        self.eos_id = 1
        self.pad_id = 0
        pre = tk.get("pre_tokenizer") or {}
        self.add_prefix_space = bool(pre.get("add_prefix_space", False))
        # 反查表：id -> piece（decode 用）
        self.id2piece = {i: p for p, i in self.vocab.items()}

    # ------------------------------ encode ------------------------------
    def _viterbi(self, piece: str) -> List[int]:
        """单个 pre-token 的最大对数似然切分（动态规划 + UNK 兜底单字）。"""
        n = len(piece)
        neg = -1e18
        best = [neg] * (n + 1)
        back: List[Tuple[int, int]] = [(-1, -1)] * (n + 1)
        best[0] = 0.0
        for i in range(n):
            if best[i] == neg:
                continue
            base = best[i]
            hi = min(i + self.maxlen, n)
            for j in range(i + 1, hi + 1):
                tid = self.vocab.get(piece[i:j])
                if tid is None:
                    continue
                sc = base + self.scores[tid]
                if sc > best[j]:
                    best[j] = sc
                    back[j] = (i, tid)
            sc_unk = base + self.unk_score
            if sc_unk > best[i + 1]:
                best[i + 1] = sc_unk
                back[i + 1] = (i, self.unk_id)
        if best[n] == neg:
            return [self.unk_id] * n
        ids: List[int] = []
        cur = n
        while cur > 0:
            prev, tid = back[cur]
            if prev < 0:
                return [self.unk_id] * n
            ids.append(tid)
            cur = prev
        ids.reverse()
        return ids

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        added = self.add_prefix_space and not text.startswith(" ")
        self._prefix_added = added
        if added:
            text = " " + text
        ids: List[int] = []
        for p in _split_pieces(text):
            mapped = "".join(B2U[b] for b in p.encode("utf-8"))
            ids.extend(self._viterbi(mapped))
        if add_special:
            ids.append(self.eos_id)
        return ids

    # ------------------------------ decode ------------------------------
    def decode(self, ids: List[int], skip_special: bool = True,
               strip_prefix_space: bool = True) -> str:
        """ByteLevel 解码。strip_prefix_space 去掉 encode 时自动添加的首个空格
        （该空格是 ByteLevel(add_prefix_space=True) 的副产物，不是用户文本）。"""
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i in (0, 1, 2) and skip_special:
                continue
            p = self.id2piece.get(i)
            if p is None:
                continue
            for ch in p:
                b = U2B.get(ch)
                if b is not None:
                    buf.append(b)
        text = buf.decode("utf-8", errors="replace")
        if (strip_prefix_space and getattr(self, "_prefix_added", False)
                and text.startswith(" ")):
            text = text[1:]
        return text

    # ------------------------------ 自检 -------------------------------
    def selfcheck(self, samples: List[str]) -> str:
        lines = ["=== tokenizer 往返自检 ===",
                 f"vocab={len(self.vocab)} maxlen={self.maxlen} "
                 f"unk_id={self.unk_id} add_prefix_space={self.add_prefix_space}"]
        ok = 0
        for s in samples:
            ids = self.encode(s)
            back = self.decode(ids)
            same = back == s
            ok += int(same)
            lines.append(f"[{'PASS' if same else 'FAIL'}] {len(s)}字符 -> {len(ids)}token "
                         f"| in={s!r} out={back!r} ids[:10]={ids[:10]}")
        lines.append(f"往返通过 {ok}/{len(samples)}")
        return "\n".join(lines)


if __name__ == "__main__":
    import os
    d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(d, "model_50m_instruct", "tokenizer.json")
    if not os.path.exists(p):
        p = r"C:\Users\Administrator\AppData\Roaming\Tencent\Marvis\User\63310B65C128F66B7C48CFE423B469AE\workspace\conv_5ef3d45527ba41debb4d1cd637fe1c9d\output\llm50m\model\tokenizer.json"
    tk = UnigramTokenizer(p)
    print(tk.selfcheck([
        "Hello world", "What is 2+2?", "Once upon a time, there was a little cat.",
        "你好，介绍一下你自己", "今天天气不错。", "Write a short story about a cat.",
        "中文 English 混排 123", "  leading spaces", "line1\nline2\ttab",
    ]))
