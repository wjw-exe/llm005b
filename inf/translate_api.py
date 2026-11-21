# -*- coding: utf-8 -*-
"""公共翻译 API 抽象层（零第三方库，仅标准库 urllib）。

设计目标：让只会在英文上工作的 0.05B 模型支持中文输入输出，而不改动模型本身。

提供方链（按顺序回退）：
  1. MyMemory        https://api.mymemory.translated.net/get  免 key，实测可用；
                     单次查询硬上限 500 字符，超长按标点切片后串行拼接。
  2. LibreTranslate  自建/公共实例，需 endpoint（多数公共实例要 key），默认关闭。

安全与可用性约束：
  * 每个请求 6 秒超时，失败立即换下一个提供方，绝不重试、绝不挂起；
  * 全链失败时降级为「原文直通」（ok=False），由调用方提示后按原文继续对话；
  * 结果写磁盘 LRU 缓存（data/translate_cache.json），命中不发请求。
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

MAX_CHUNK = 480          # MyMemory 硬上限 500 字符，留 20 字符余量
DEFAULT_TIMEOUT = 6.0
CACHE_LIMIT = 2000
CJK_MIN_RATIO = 0.15     # CJK 字符占比达到该阈值即判定为中文

_UA = {"User-Agent": "curl/8.4.0"}   # MyMemory 对空 UA 偶发 403
_LANG = {"zh": "zh-CN", "en": "en"}
_PUNCT = "。！？；，、：…\n.!?;,:"   # 切句标点（中英混排）


class TranslateError(Exception):
    """单个提供方翻译失败（用于触发换源）。"""


# --------------------------------------------------------------- 分片工具
def split_chunks(text: str, limit: int = MAX_CHUNK) -> List[str]:
    """按标点把长文本切成 <= limit 字符的片段，保序、不丢字符。

    先按标点累积，单句本身超长时退化为按字符硬切（保证一定能发出去）。
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    chunks, buf = [], ""
    for ch in text:
        buf += ch
        if len(buf) >= limit or (ch in _PUNCT and len(buf) >= limit // 2):
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    out: List[str] = []
    for c in chunks:                     # 单句仍超限时硬切
        while len(c) > limit:
            out.append(c[:limit])
            c = c[limit:]
        if c.strip():
            out.append(c)
    return out


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=dict(_UA, **{
        "Content-Type": "application/json"}))
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310
        return json.loads(r.read().decode("utf-8"))


# --------------------------------------------------------------- 提供方
class Provider:
    """翻译提供方基类：子类实现 translate(text, src, dst) -> str。"""

    name = "base"
    note = ""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def available(self) -> bool:
        return True

    def translate(self, text: str, src: str, dst: str) -> str:
        raise NotImplementedError


class MyMemoryProvider(Provider):
    """MyMemory 免费接口：免 key、单次 <=500 字符、中文走 zh-CN。"""

    name = "mymemory"
    note = "免 key，公共免费接口（限速约 1 请求/秒，单次 500 字符）"
    URL = "https://api.mymemory.translated.net/get"

    def _one(self, text: str, lp: str) -> str:
        url = "%s?q=%s&langpair=%s" % (self.URL, urllib.parse.quote(text),
                                       urllib.parse.quote(lp))
        try:
            d = _get_json(url, self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TranslateError("网络/超时: %s" % str(exc)[:60]) from exc
        except ValueError as exc:
            raise TranslateError("返回非 JSON: %s" % str(exc)[:60]) from exc
        st = str(d.get("responseStatus"))
        out = ((d.get("responseData") or {}).get("translatedText") or "").strip()
        if st != "200":
            raise TranslateError("状态 %s: %s" % (st, str(d.get("responseDetails"))[:80]))
        if not out or "QUERY LENGTH LIMIT" in out.upper():
            raise TranslateError("返回内容异常: %s" % out[:60])
        return out

    def translate(self, text: str, src: str, dst: str) -> str:
        lp = "%s|%s" % (_LANG.get(src, src), _LANG.get(dst, dst))
        parts = []
        for c in split_chunks(text):
            parts.append(self._one(c, lp))     # 串行：公共接口限速，避免 429
        return " ".join(parts).strip()


class LibreTranslateProvider(Provider):
    """LibreTranslate 实例（需 endpoint；多数公共实例需 api key 或登录）。"""

    name = "libretranslate"
    note = "需自备 endpoint（公共实例多需 key）"

    def __init__(self, endpoint: str = "", key: str = "",
                 timeout: float = DEFAULT_TIMEOUT):
        super().__init__(timeout)
        self.endpoint = (endpoint or "").rstrip("/")
        self.key = key

    def available(self) -> bool:
        return bool(self.endpoint)

    def translate(self, text: str, src: str, dst: str) -> str:
        if not self.endpoint:
            raise TranslateError("未配置 endpoint")
        payload = {"q": text, "source": src, "target": dst, "format": "text"}
        if self.key:
            payload["api_key"] = self.key
        try:
            d = _post_json(self.endpoint + "/translate", payload, self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TranslateError("网络/超时: %s" % str(exc)[:60]) from exc
        except ValueError as exc:
            raise TranslateError("返回非 JSON: %s" % str(exc)[:60]) from exc
        out = (d.get("translatedText") or "").strip()
        if not out:
            raise TranslateError("返回为空: %s" % str(d)[:60])
        return out


# --------------------------------------------------------------- 缓存
class LRUDiskCache:
    """进程内 OrderedDict + 磁盘 JSON 的 LRU 翻译缓存，命中不发起请求。"""

    def __init__(self, path: str, limit: int = CACHE_LIMIT):
        self.path = path
        self.limit = limit
        self.data: "OrderedDict[str, str]" = OrderedDict()
        self.hit = 0
        self.miss = 0
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, str):
                        self.data[k] = v
        except (OSError, ValueError):
            self.data = OrderedDict()

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(dict(self.data), f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass   # 缓存写失败不影响对话

    @staticmethod
    def key(src: str, dst: str, text: str) -> str:
        return "%s>%s|%s" % (src, dst, text)

    def get(self, k: str) -> Optional[str]:
        if k in self.data:
            self.data.move_to_end(k)
            self.hit += 1
            return self.data[k]
        self.miss += 1
        return None

    def put(self, k: str, v: str) -> None:
        self.data[k] = v
        self.data.move_to_end(k)
        while len(self.data) > self.limit:
            self.data.popitem(last=False)
        self._save()

    def __len__(self) -> int:
        return len(self.data)


# --------------------------------------------------------------- 门面
class Translator:
    """提供方链 + 语言判定 + 缓存的对外门面。"""

    def __init__(self, providers: Optional[List[Provider]] = None,
                 cache_path: str = "", timeout: float = DEFAULT_TIMEOUT,
                 log=None):
        self.log = log or (lambda *a, **k: None)
        self.providers = providers or [MyMemoryProvider(timeout)]
        self.cache = LRUDiskCache(cache_path) if cache_path else None
        self.requests = 0
        self.failures = 0

    # -------------------------------------------------- 语言判定
    @staticmethod
    def detect(text: str) -> str:
        """按 CJK 字符占比判定语种：返回 'zh' 或 'en'。"""
        letters = [ch for ch in text if not ch.isspace()]
        if not letters:
            return "en"
        cjk = sum(1 for ch in letters
                  if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf")
        return "zh" if cjk / len(letters) >= CJK_MIN_RATIO else "en"

    def direction(self, text: str) -> Tuple[str, str]:
        """中文输入 → ('zh','en')；英文输入 → ('en','zh')。"""
        return ("zh", "en") if self.detect(text) == "zh" else ("en", "zh")

    @property
    def names(self) -> str:
        ns = [p.name for p in self.providers if p.available()]
        return ",".join(ns) if ns else "无（所有提供方均未启用）"

    # -------------------------------------------------- 翻译
    def translate(self, text: str, src: str, dst: str) -> Dict[str, object]:
        """返回 {ok, text, provider, cached, seconds, error}；失败降级为原文。"""
        text = (text or "").strip()
        res = {"ok": False, "text": text, "provider": "-", "cached": False,
               "seconds": 0.0, "error": ""}
        if not text:
            res["ok"] = True
            return res
        ck = LRUDiskCache.key(src, dst, text)
        t0 = time.perf_counter()
        if self.cache is not None:
            hit = self.cache.get(ck)
            if hit is not None:
                res.update(ok=True, text=hit, provider="cache", cached=True,
                           seconds=time.perf_counter() - t0)
                return res
        errs = []
        for p in self.providers:
            if not p.available():
                errs.append("%s: 未启用" % p.name)
                continue
            self.requests += 1
            try:
                out = p.translate(text, src, dst)
                res.update(ok=True, text=out, provider=p.name,
                           seconds=time.perf_counter() - t0)
                if self.cache is not None:
                    self.cache.put(ck, out)
                return res
            except TranslateError as exc:
                errs.append("%s: %s" % (p.name, exc))
                self.log("[翻译] %s 失败，尝试下一个来源：%s" % (p.name, exc))
            except Exception as exc:     # noqa: BLE001  防御：任何异常都不许中断对话
                errs.append("%s: %s" % (p.name, type(exc).__name__))
                self.log("[翻译] %s 异常：%s" % (p.name, exc))
        self.failures += 1
        res.update(seconds=time.perf_counter() - t0, error="; ".join(errs))
        return res

    def status(self) -> str:
        c = self.cache
        return ("提供方链: %s | 请求 %d 次 / 失败 %d 次 | 缓存 %d 条（命中 %d）"
                % (self.names or "无", self.requests, self.failures,
                   len(c) if c else 0, c.hit if c else 0))
