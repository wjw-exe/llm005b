# -*- coding: utf-8 -*-
"""0.05B 指令微调模型对话前端（手写算子真实推理通路）。

与 chat.py 的原「自训练模型」通路并列：本文件只负责公开 50M 指令模型
(hayder86al/tinystories-50m-instruct, MiniGPT 架构) 的交互界面与多轮编排，
所有算子仍来自工程内 hand-written numpy 实现（inf/minigpt_numpy.py、ops.py），
不依赖 torch / transformers / onnx。
"""
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from minigpt_numpy import KVCache, MiniGPTInstruct  # noqa: E402
from unigram_tok import _bytes_to_unicode  # noqa: E402

try:  # 翻译增强为可选能力：导入失败时英文直连模式照常可用
    from translate_api import LibreTranslateProvider, MyMemoryProvider, Translator  # noqa: E402
    _TR_IMPORT_ERR = ""
except Exception as _e:  # noqa: BLE001
    Translator = None  # type: ignore
    MyMemoryProvider = None  # type: ignore
    LibreTranslateProvider = None  # type: ignore
    _TR_IMPORT_ERR = "%s: %s" % (type(_e).__name__, _e)

B2U = _bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}
MODEL_DIR = os.path.join(ROOT, "model_50m_instruct")
WEIGHT_FILE = os.path.join(MODEL_DIR, "model.safetensors")

STOP_STRINGS = ("\nUser:", "\nuser:", "\n\nUser:", "\nAssistant:")
SPECIAL_IDS = (0, 1, 2)  # <pad> <eos> <unk>


def tok_bytes(tk, tid: int) -> bytes:
    """把 token id 还原成原始字节（ByteLevel 反映射，用于 UTF-8 增量解码）。"""
    piece = tk.id2piece.get(tid, "")
    if not piece:
        return b""
    out = bytearray()
    for ch in piece:
        b = U2B.get(ch)
        if b is None:
            return b""  # 特殊 token（<pad>/<eos>/<unk>）不产出字节
        out.append(b)
    return bytes(out)


def human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return ("%d%s" % (n, unit)) if unit == "" else ("%.2f%s" % (n, unit))
        n /= 1000.0
    return "%.2fT" % n


def rss_mb() -> Tuple[float, float]:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                      ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 1048576.0, pmc.PeakWorkingSetSize / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0, 0.0


class InstructChat:
    """公开 50M 指令微调模型的对话循环（多轮、流式、模型可切换）。"""

    TURN_FMT = "User: %s\nAssistant: %s\n\n"
    ASK_FMT = "User: %s\nAssistant:"

    def __init__(self, model_dir: str = MODEL_DIR, temp: float = 0.5,
                 top_k: int = 40, top_p: float = 0.95, max_new: int = 120,
                 mode: str = "single", verbose: bool = True,
                 translate: bool = False, provider: str = "",
                 lib_endpoint: str = "", lib_key: str = ""):
        if not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
            print("=" * 62)
            print(" 未找到指令模型权重：%s" % os.path.join(model_dir, "model.safetensors"))
            print(" 请确认 model_50m_instruct/ 目录完整（config + safetensors + tokenizer）。")
            print("=" * 62)
            raise SystemExit(2)
        self.m = MiniGPTInstruct(model_dir, verbose=verbose)
        self.tok = self.m.tok
        self.ctx = self.m.ctx
        self.eos_id = self.m.eos_id
        self.model_dir = model_dir
        self.weight_mb = os.path.getsize(os.path.join(model_dir, "model.safetensors")) / 1048576.0
        self.param_count = int(sum(int(v.size) for v in self.m.p.values()))
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.max_new = max_new
        self.history: List[Tuple[str, str]] = []
        self.turns = 0
        self.mode = mode if mode in ("single", "multi") else "single"
        self.warned_zh = False
        self.total_gen_tokens = 0
        self.total_gen_time = 0.0
        self.budget = max(16, self.ctx - self.max_new - 8)

        # ---- 中文增强（公共翻译 API）状态 ----
        self.provider_arg = provider or "mymemory"
        self.lib_endpoint = lib_endpoint or os.environ.get("LIBRETRANSLATE_URL", "")
        self.lib_key = lib_key or os.environ.get("LIBRETRANSLATE_KEY", "")
        self.tr = None
        self.translate_on = False
        self.show_raw = True          # 是否显示模型英文原文（/raw off 可关）
        self.tr_stats = {"zh_turns": 0, "ok_in": 0, "ok_out": 0, "fail": 0,
                         "sec_in": 0.0, "sec_out": 0.0}
        if translate:
            self.enable_translate(True)

    # ------------------------------------------------------ 翻译增强控制
    def build_translator(self):
        names = [s.strip().lower() for s in
                 self.provider_arg.replace("+", ",").replace("/", ",").split(",") if s.strip()]
        provs = []
        for n in names:
            if n in ("mymemory", "mm", "default", "free"):
                provs.append(MyMemoryProvider())
            elif n in ("libretranslate", "libre", "lt", "local"):
                provs.append(LibreTranslateProvider(self.lib_endpoint, self.lib_key))
            else:
                print("（未知翻译提供方 %s，已忽略）" % n)
        if not provs:
            provs = [MyMemoryProvider()]
        return Translator(providers=provs,
                          cache_path=os.path.join(ROOT, "data", "translate_cache.json"),
                          log=print)

    def enable_translate(self, on: bool) -> bool:
        if not on:
            self.translate_on = False
            return True
        if Translator is None:
            print("（翻译模块不可用：%s）" % _TR_IMPORT_ERR)
            return False
        if self.tr is None:
            try:
                self.tr = self.build_translator()
            except Exception as exc:  # noqa: BLE001
                print("（翻译器初始化失败：%s: %s）" % (type(exc).__name__, exc))
                return False
        self.translate_on = True
        return True

    # -------------------------------------------------------- 上下文拼装
    def build_prompt(self, user_text: str) -> Tuple[List[int], int]:
        """返回 (prompt_ids, 丢弃的旧轮数)；超长自动截断最旧轮次。

        模型仅在 `User: ... / Assistant: ...` 单轮格式上做过指令微调
        （官方 model card 明确 single-turn only），因此默认 single 模式只发
        当前问题；multi 模式会拼接历史轮次（模型会明显退化，但上下文与
        1024 截断逻辑真实可用）。
        """
        turns = list(self.history) if self.mode == "multi" else []
        dropped = 0
        while True:
            txt = "".join(self.TURN_FMT % (u, a) for u, a in turns)
            txt += self.ASK_FMT % user_text
            ids = self.tok.encode(txt)
            if len(ids) <= self.budget or not turns:
                break
            turns = turns[1:]
            dropped += 1
        if len(ids) > self.budget:  # 单轮本身超长：保留尾部（含提问）
            ids = ids[-self.budget:]
        return ids, dropped

    # ------------------------------------------------------------ 生成
    def reply(self, prompt_ids: List[int], on_text=None) -> dict:
        m = self.m
        window = list(prompt_ids)
        t0 = time.perf_counter()
        logits, cache = m.forward(window, None, 0)
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        used = len(window)
        t_gen0 = time.perf_counter()
        t_first = None
        out_ids: List[int] = []
        buf = b""
        shown = 0
        for _ in range(self.max_new):
            if used >= self.ctx:  # 窗口滑出最旧 token 并重建 cache
                window = window[-(self.ctx // 2):]
                logits, cache = m.forward(window, None, 0)
                used = len(window)
            tok = m._sample(logits[0, -1], self.temp, self.top_k, self.top_p)
            if t_first is None:
                t_first = (time.perf_counter() - t_gen0) * 1000.0
            if tok in SPECIAL_IDS:
                break
            out_ids.append(tok)
            used += 1
            buf += tok_bytes(self.tok, tok)
            text = None
            try:
                text = buf.decode("utf-8")
            except UnicodeDecodeError:
                text = None  # 半个汉字，继续等字节
            if text is not None:
                new = text[shown:]
                if new:
                    if on_text is not None:
                        on_text(new)
                    shown = len(text)
                if any(s in text for s in STOP_STRINGS):
                    break
            window.append(tok)
            logits, cache = m.forward([tok], cache, used - 1)
        gen_s = max(1e-9, time.perf_counter() - t_gen0)
        full = buf.decode("utf-8", errors="ignore")
        for s in STOP_STRINGS:
            if s in full:
                full = full.split(s)[0]
        n = len(out_ids)
        self.total_gen_tokens += n
        self.total_gen_time += gen_s
        return {
            "text": full.strip(),
            "n_tokens": n,
            "gen_seconds": gen_s,
            "tok_per_s": (n / gen_s) if n else 0.0,
            "first_token_ms": t_first if t_first is not None else -1.0,
            "prefill_ms": prefill_ms,
            "prompt_tokens": len(prompt_ids),
            "used": used,
        }

    def ask(self, line: str) -> None:
        """一轮完整对话：中文输入 → 译英 → 模型推理 → 英文回答 → 回译中文。"""
        use_tr = self.translate_on and self.tr is not None
        is_zh = bool(use_tr) and self.tr.detect(line) == "zh"
        model_input, tr_in = line, None
        if is_zh:
            print("[译入] 正在把中文输入翻成英文 ...")
            tr_in = self.tr.translate(line, "zh", "en")
            if tr_in["ok"]:
                model_input = tr_in["text"]
                self.tr_stats["zh_turns"] += 1
                self.tr_stats["ok_in"] += 1
                self.tr_stats["sec_in"] += tr_in["seconds"]
                print("[译入] %s" % model_input)
                print("  └ 提供方 %s%s，耗时 %.2fs"
                      % (tr_in["provider"], "（磁盘缓存命中，未发请求）" if tr_in["cached"] else "",
                         tr_in["seconds"]))
            else:
                self.tr_stats["fail"] += 1
                print("[译入失败] 翻译服务不可用，已降级为原文直送模型"
                      "（模型无中文能力，回答大概率无意义）")
                print("  └ 原因：%s" % str(tr_in["error"])[:200])
        elif not use_tr and not self.warned_zh and any(ord(ch) > 127 for ch in line):
            self.warned_zh = True
            print("（提示：该权重仅用英文 TinyStories 语料训练，词表无中文字符，"
                  "中文以 UTF-8 字节碎片表示；可用 /translate on 开启公共翻译 API 中文增强）")

        prompt_ids, dropped = self.build_prompt(model_input)
        if dropped:
            print("（上下文超限，已自动丢弃最旧的 %d 轮）" % dropped)
        stream = (not is_zh) or self.show_raw
        print("Assistant(EN)> " if is_zh else "Assistant> ", end="", flush=True)
        try:
            r = self.reply(prompt_ids,
                           on_text=(lambda s: (sys.stdout.write(s), sys.stdout.flush()))
                           if stream else None)
        except KeyboardInterrupt:
            print("\n[已中断本次生成]")
            return
        if not r["text"]:
            print("（模型只输出了空内容，可尝试 /temp 0.7 或 /reset）")

        tr_out = None
        if is_zh:
            if r["text"]:
                print("\n[译出] 正在把英文回答翻回中文 ...")
                tr_out = self.tr.translate(r["text"], "en", "zh")
                if tr_out["ok"]:
                    self.tr_stats["ok_out"] += 1
                    self.tr_stats["sec_out"] += tr_out["seconds"]
                    print("[中文回显] %s" % tr_out["text"])
                    print("  └ 提供方 %s%s，耗时 %.2fs"
                          % (tr_out["provider"],
                             "（磁盘缓存命中，未发请求）" if tr_out["cached"] else "",
                             tr_out["seconds"]))
                    if self.show_raw:
                        print("[英文原文] %s" % r["text"])
                else:
                    self.tr_stats["fail"] += 1
                    print("[译出失败] 回译不可用，以下为模型英文原文：")
                    print("[英文原文] %s" % r["text"])
                    print("  └ 原因：%s" % str(tr_out["error"])[:200])
            else:
                print("[中文回显] （模型输出为空，无需回译）")

        if is_zh:
            t_in = tr_in["seconds"] if (tr_in and tr_in["ok"]) else 0.0
            t_out = tr_out["seconds"] if (tr_out and tr_out["ok"]) else 0.0
            print("[链路] 译入 %.2fs(%s) | 推理 %.2fs / %d tok (%.2f tok/s) | 译出 %.2fs(%s) "
                  "| 合计 %.2fs"
                  % (t_in, (tr_in["provider"] if tr_in else "-"),
                     r["gen_seconds"], r["n_tokens"], r["tok_per_s"],
                     t_out, (tr_out["provider"] if tr_out else "-"),
                     t_in + r["gen_seconds"] + t_out))
        else:
            print("\n[%d tok / %.2fs = %.2f tok/s | prefill %.0f ms + 首 token %.2fs | 上下文 %d/%d | 峰值 %.0f MB]"
                  % (r["n_tokens"], r["gen_seconds"], r["tok_per_s"], r["prefill_ms"],
                     r["first_token_ms"] / 1000.0, r["used"], self.ctx, rss_mb()[1]))
        self.history.append((model_input, r["text"]))
        self.turns += 1

    # ------------------------------------------------------------ 界面
    def banner(self):
        rss, peak = rss_mb()
        print("=" * 64)
        print(" llm005b 对话助手 · 0.05B 指令微调模型（公开权重 + 手写算子）")
        print(" 模型: hayder86al/tinystories-50m-instruct (MiniGPT 自定义架构)")
        print(" 结构: layers=%d heads=%d embd=%d ctx=%d vocab=%d 权重共享=head/token_emb"
              % (self.m.n_layer, self.m.n_head, self.m.n_embd, self.ctx, self.m.vocab))
        print(" 权重: model_50m_instruct/model.safetensors (%.1f MB)  参数: %s"
              % (self.weight_mb, human(self.param_count)))
        print(" 推理: 手写 numpy 算子 + 预分配 KV Cache 增量解码 + safetensors 自解析")
        print(" 采样: temp=%.2f top_k=%d top_p=%.2f max_new=%d"
              % (self.temp, self.top_k, self.top_p, self.max_new))
        print(" 上下文模式: %s（single=仅发当前问题，贴合模型单轮训练分布；"
              "multi=拼接历史，超 1024 自动丢弃最旧轮次）" % self.mode)
        print(" 已知边界: 50M 参数 / TinyStories 英文童话语料，官方卡明确仅单轮、"
              "不保证事实准确性；中文与算术基本不可用")
        print(" 中文增强: %s" % ("开启（提供方链 %s；中文输入→译英→模型→回译中文）"
                                 % (self.tr.names if self.tr else self.provider_arg)
                                 if self.translate_on else
                                 "关闭（纯英文直连；/translate on 开启公共翻译 API 中文增强）"))
        print(" 英文原文: %s（/raw on|off 切换；需联网，公共接口较慢，约 2-4 秒/次翻译）"
              % ("显示" if self.show_raw else "隐藏"))

        if rss > 0:
            print(" 内存: 当前 %.0f MB / 峰值 %.0f MB" % (rss, peak))
        print(" 输入问题后回车；/help 查看命令；Ctrl+C 退出")
        print("=" * 64)

    def stats(self):
        rss, peak = rss_mb()
        speed = (self.total_gen_tokens / self.total_gen_time) if self.total_gen_time > 0 else 0.0
        print("---- 运行统计 ----")
        print(" 当前模型        : 0.05B 指令微调 (tinystories-50m-instruct)")
        print(" 对话轮数        : %d" % self.turns)
        print(" 已生成 token    : %d" % self.total_gen_tokens)
        print(" 平均生成速度    : %.2f tok/s（%.1f ms/token）"
              % (speed, (1000.0 / speed) if speed > 0 else 0.0))
        print(" 上下文上限      : %d token（超限自动丢弃最旧轮次）" % self.ctx)
        print(" 当前上下文占用  : %d token" % self.current_len())
        print(" 参数总量        : %s（去重后实际进图）" % human(self.param_count))
        st = self.tr_stats
        if self.translate_on and self.tr is not None:
            print(" 中文增强        : 开启（提供方链 %s）" % self.tr.names)
            print(" 中文轮次        : %d（译入成功 %d / 回译成功 %d / 失败 %d）"
                  % (st["zh_turns"], st["ok_in"], st["ok_out"], st["fail"]))
            print(" 翻译累计耗时    : 译入 %.2fs / 译出 %.2fs" % (st["sec_in"], st["sec_out"]))
            print(" 缓存与请求      : %s" % self.tr.status())
        else:
            print(" 中文增强        : 关闭（纯英文直连）")

        if rss > 0:
            print(" 内存            : 当前 %.0f MB / 峰值 %.0f MB" % (rss, peak))

    def current_len(self) -> int:
        txt = "".join(self.TURN_FMT % (u, a) for u, a in self.history)
        return len(self.tok.encode(txt)) if txt else 0

    def run(self) -> object:
        self.banner()
        while True:
            try:
                line = input("\nYou> ").strip()
            except KeyboardInterrupt:
                print("\n已退出。")
                self.stats()
                return 0
            except EOFError:
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                cmd = line.split()
                name = cmd[0].lower()
                arg = cmd[1] if len(cmd) > 1 else ""
                if name in ("/exit", "/quit", "/q"):
                    self.stats()
                    return 0
                elif name in ("/reset", "/clear"):
                    self.history = []
                    print("上下文已清空。")
                elif name == "/model":
                    if arg in ("", "instruct", "0.05b", "new"):
                        print("当前模型：0.05B 指令微调（tinystories-50m-instruct）")
                        print("可用：/model scratch 切换到工程自训练模型")
                    elif arg in ("scratch", "self", "old"):
                        print("正在切换到自训练模型（ckpt/final.npz）...")
                        return "scratch"
                    else:
                        print("用法：/model [instruct|scratch]")
                elif name == "/temp":
                    try:
                        self.temp = float(arg)
                        print("温度 = %.2f" % self.temp)
                    except ValueError:
                        print("用法：/temp 0.8")
                elif name == "/topk":
                    try:
                        self.top_k = int(arg)
                        print("top_k = %d" % self.top_k)
                    except ValueError:
                        print("用法：/topk 40")
                elif name == "/topp":
                    try:
                        self.top_p = float(arg)
                        print("top_p = %.2f" % self.top_p)
                    except ValueError:
                        print("用法：/topp 0.9")
                elif name == "/mode":
                    if arg in ("single", "multi"):
                        self.mode = arg
                        print("上下文模式 = %s" % self.mode)
                        if arg == "multi":
                            print("提示：该模型仅在单轮格式上微调，多轮历史会让回答质量下降。")
                    else:
                        print("当前模式：%s；用法：/mode single|multi" % self.mode)
                elif name == "/translate":
                    a = arg.strip().lower()
                    if a in ("on", "1", "true", "yes"):
                        if self.enable_translate(True):
                            print("翻译模式 = on（中文输入→译英→模型→回译中文）")
                            print("提供方链: %s | %s" % (self.tr.names, self.tr.status()))
                        else:
                            print("翻译模式 = off（未能开启，原因见上）")
                    elif a in ("off", "0", "false", "no"):
                        self.enable_translate(False)
                        print("翻译模式 = off（纯英文直连，中文输入不再翻译）")
                    elif a in ("", "status"):
                        if self.translate_on and self.tr is not None:
                            print("翻译模式 = on | %s" % self.tr.status())
                            print("缓存文件: data/translate_cache.json | 英文原文显示: %s"
                                  % ("on" if self.show_raw else "off"))
                        else:
                            print("翻译模式 = off（纯英文直连）")
                        print("用法：/translate on|off|status；/translate provider mymemory"
                              " 或 mymemory,libretranslate")
                    elif a.startswith("provider"):
                        p = arg[len("provider"):].strip()
                        if p:
                            self.provider_arg = p
                            self.tr = None
                            if self.translate_on:
                                self.enable_translate(True)
                            print("提供方链已设为: %s%s"
                                  % (self.provider_arg,
                                     "（已重建）" if self.translate_on else "（下次开启时生效）"))
                        else:
                            print("当前提供方链: %s；用法：/translate provider mymemory"
                                  % (self.tr.names if self.tr else self.provider_arg))
                    else:
                        print("用法：/translate on|off|status|provider <names>")
                elif name == "/raw":
                    a = arg.strip().lower()
                    if a in ("on", "1", "true", "yes"):
                        self.show_raw = True
                        print("英文原文显示 = on（中文轮同时显示模型英文原文）")
                    elif a in ("off", "0", "false", "no"):
                        self.show_raw = False
                        print("英文原文显示 = off（中文轮只显示中文回显，英文原文仍可 /raw on 查看）")
                    else:
                        print("英文原文显示 = %s；用法：/raw on|off"
                              % ("on" if self.show_raw else "off"))
                elif name == "/stats":
                    self.stats()
                elif name == "/help":
                    print("命令：/reset /temp <v> /topk <n> /topp <p> /mode single|multi "
                          "/model [instruct|scratch] /translate on|off|status "
                          "/raw on|off /stats /help /exit")
                    print("  /translate on  开启公共翻译 API 中文增强（中文输入→译英→模型→回译中文）")
                    print("  /translate off 关闭，回到纯英文直连；status 查看提供方链与缓存命中")
                    print("  /raw on|off    中文轮是否同时显示模型英文原文")

                    print("  /mode single 仅发当前问题（推荐，贴合单轮训练分布）；"
                          "multi 拼接历史，超 %d token 自动丢弃最旧轮次" % self.m.ctx)
                    print("  /model scratch 切到工程自训练模型；/model instruct 切回 0.05B 指令模型")
                else:
                    print("未知命令，/help 查看。")
                continue

            self.ask(line)



if __name__ == "__main__":
    sys.exit(InstructChat().run())
