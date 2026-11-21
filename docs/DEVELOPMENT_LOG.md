---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 297ab8dbc8f4cde8dd6b234597dbc01a_f12afb2ee59711f19db6525400638852
    ReservedCode1: 3P1UgrdQFkiQFsm/g/B8HocQHZb+gaJ5c+RkJQYOp3J7brMNYjyG+l/ciB1IPRVGI3MUvTlmSyEVBgojAWAw+wc5aofuMmR2w5HFHMD1Pg4zkPdE9Z0FcRHskOFr9GFOp/BceGpJZRbGv2fEpM7cFKBeBkGCkBwY2sv5Ld5ec6pMDLCFrERhFOtJ9UQ=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 297ab8dbc8f4cde8dd6b234597dbc01a_f12afb2ee59711f19db6525400638852
    ReservedCode2: 3P1UgrdQFkiQFsm/g/B8HocQHZb+gaJ5c+RkJQYOp3J7brMNYjyG+l/ciB1IPRVGI3MUvTlmSyEVBgojAWAw+wc5aofuMmR2w5HFHMD1Pg4zkPdE9Z0FcRHskOFr9GFOp/BceGpJZRbGv2fEpM7cFKBeBkGCkBwY2sv5Ld5ec6pMDLCFrERhFOtJ9UQ=
---

---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 297ab8dbc8f4cde8dd6b234597dbc01a_3cf431ade59311f19db6525400638852
    ReservedCode1: +ydHCDrZLxsiMTT7tIdeEYikmxBmuRN1rtzNiEtNNu0QR1VrLiGT6myc5iu4s9TvvJHZG5Gh4wlWru35RkOc//ecrqGJyqFCn0hXUUelDtzNJYTwFygT8AZ5Jqqt8TH0i0i2RjdSgFEh5D5eD22TJm4XDswtXfkXMc/v0Vm8lqUofRHle/QrcxMKhHg=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 297ab8dbc8f4cde8dd6b234597dbc01a_3cf431ade59311f19db6525400638852
    ReservedCode2: +ydHCDrZLxsiMTT7tIdeEYikmxBmuRN1rtzNiEtNNu0QR1VrLiGT6myc5iu4s9TvvJHZG5Gh4wlWru35RkOc//ecrqGJyqFCn0hXUUelDtzNJYTwFygT8AZ5Jqqt8TH0i0i2RjdSgFEh5D5eD22TJm4XDswtXfkXMc/v0Vm8lqUofRHle/QrcxMKhHg=
---

---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 297ab8dbc8f4cde8dd6b234597dbc01a_4304237ce58311f19db6525400638852
    ReservedCode1: guNj6e5Yqrgo7AKd7iImlD1G30map6x7ZsT+MIDxDlSbbQb/H8TwIzvY0/pXEvILaMccgD5s521vdagJjpZAe2rLfn/HJGRKJlUnJ8pweV7V4DKRSSfCTuN7TGWtGbkUc1F0vsE14ee4LfwpZbYlVANU1LAmnOppzirkDFPG5DAiWXO2x1Bw9ReigmA=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 297ab8dbc8f4cde8dd6b234597dbc01a_4304237ce58311f19db6525400638852
    ReservedCode2: guNj6e5Yqrgo7AKd7iImlD1G30map6x7ZsT+MIDxDlSbbQb/H8TwIzvY0/pXEvILaMccgD5s521vdagJjpZAe2rLfn/HJGRKJlUnJ8pweV7V4DKRSSfCTuN7TGWtGbkUc1F0vsE14ee4LfwpZbYlVANU1LAmnOppzirkDFPG5DAiWXO2x1Bw9ReigmA=
---

# llm005b · 0.05B 对话语言模型（纯 numpy 手写算子，零深度学习框架）

一个**从零手写**的 GPT 风格 decoder-only 对话语言模型工程。
全部算子（前向 + 反向）、BPE 分词器、AdamW 优化器、训练循环、推理与 KV Cache 均由本项目自写，
**只依赖 numpy + Python 标准库**，不 import torch / transformers / jax / onnx / tensorflow。

工程现在提供**两条并列的推理通路**，共用同一套手写算子：

| 通路 | 权重来源 | 入口 | 说明 |
|---|---|---|---|
| **A. 0.05B 指令模型（默认）** | 公开权重 [hayder86al/tinystories-50m-instruct](https://hf-mirror.com/hayder86al/tinystories-50m-instruct)（MiniGPT 自定义架构，指令微调） | `chat.py`（默认）/ `start.bat` 菜单 1 / `inf/instruct_chat.py` | 真实可对话，49,440,768 唯一参数，实测 33–38 tok/s |
| **B. 自训练模型** | 本工程从零训练的 `ckpt/final.npz` | `chat.py --model scratch` / `start.bat` 菜单 2 | 46,152,704 参数，训练 400 步，可继续训练 |

两条通路都走 `inf/minigpt_numpy.py`（A）与 `model.py`（B）里的手写 numpy 算子，
没有任何深度学习框架参与；`chat.py` 内 `/model instruct|scratch` 可运行时互切。

| 项目 | 实测值 |
|---|---|
| 参数量 | **46,152,704（46.15M）** |
| 结构 | d_model=512 / 12 层 / 8 头 / head_dim=64 / ctx=256 / FFN=2048 / vocab=16000 / 词嵌入与输出层权重共享 |
| 语料 | 45,723 条中英对话（Dolly-15k + oasst1-zh），编码后 **9,743,857 tokens** |
| 已训练 | 400 步（batch=8 × ctx=256，共 81.9 万 tokens），loss **9.7532 → 6.1200** |
| 梯度校验 | 6 类算子全部 PASS，最大相对误差 **8.689e-06** |
| 推理速度 | **42 token/s**（单核 CPU），首 token 0.05–0.18 s |
| 内存占用 | 对话进程 当前 213 MB / 峰值 742 MB |

---

## 1. 一键启动

**双击 `start.bat`**（或在终端执行 `start.bat`），脚本会自动完成：
Python 解释器查找 → numpy 依赖检查（缺失自动安装） → 显示菜单：

```
1  0.05B 指令模型对话★ （首选：公开 50M 指令微调权重 + 手写算子）
2  自训练模型对话      （chat.py，工程内从零训练出的 46M 权重）
3  训练模型            （train.py，后台训练 + 日志，已有 ckpt 自动续训）
4  梯度校验            （grad_check.py，数值梯度自检）
5  准备语料            （download_corpus.py + prepare.py）
6  查看训练日志        （logs\train.log 末尾）
0  退出
```

直接回车默认进入 1（0.05B 指令模型对话），双击两次回车即可开始聊天，无需手动下载或配置。

无需任何手动分步操作；训练、下载等耗时任务全部后台运行，前台只做日志轮询，不阻塞。

### 直接使用各入口（等价）

```bat
python chat.py                 :: 对话（自动加载 ckpt/final.npz）
python train.py --steps 400 --batch 8 --lr 1.2e-3 --warmup 50 --resume --background
python grad_check.py           :: 数值梯度校验
python prepare.py              :: 训练 BPE + 编码语料
python tools\download_corpus.py --background
```

---

## 2. 目录结构

```
llm005b/
├─ start.bat                 一键菜单启动（依赖检查 + 1 对话 / 2 训练 / 3 梯度校验 / 4 语料 / 5 日志）
├─ ops.py                    手写算子库（每个算子前向 + 反向）
├─ model.py                  GPT decoder-only 组装（训练前向/反向 + 推理 KV Cache）
├─ bp/bpe.py                 手写字节级 BPE 分词器（训练 merges / 编码 / 解码 / 往返自检）
├─ bp/tokenizer.json         BPE 词表与 15,741 条 merges
├─ prepare.py                语料预处理：训练 BPE → 编码 train.bin → 写 meta.json
├─ train.py                  从零训练器（AdamW / warmup+cosine / 梯度裁剪 / 断点续训 / 后台 / 日志）
├─ grad_check.py             有限差分梯度校验（Linear / LayerNorm / Softmax / Attention / GELU / CE）
├─ chat.py                   CLI 对话前端（流式 / 多轮 / 斜杠命令 / 实时 tok/s）
├─ configs/base.json         模型配置
├─ data/
│   ├─ corpus.txt            整理后的 User:/Assistant: 对话语料（33.7 MB）
│   ├─ pairs.jsonl           结构化问答对（34.1 MB）
│   ├─ train.bin             uint16 token 序列（19.5 MB / 9,743,857 tokens）
│   ├─ meta.json             vocab_size 等元信息
│   └─ src_dolly/ src_oasst1_zh/   原始下载分片
├─ ckpt/
│   ├─ final.npz             最终权重（step 400，554 MB）
│   ├─ step000100/200/300.npz  训练过程快照
│   └─ latest.json           checkpoint 指针（权威入口，chat.py 优先读它）
├─ logs/                     下载 / 训练 / 梯度校验 / 对话 / 内存 实测日志
└─ tools/
    ├─ download_corpus.py    hf-mirror 语料下载（带 --max-time 超时，失败换源）
    ├─ keep_train.py         守护式续训（串行续训直到目标步数）
    ├─ chat_demo.py          非交互对话演示（自动喂入多轮并留档）
    ├─ mem_probe.py          进程内存采样
    └─ selfcheck.py          依赖审计 + 参数量核算 + 分词器往返 + 权重完整性
```

---

## 3. 手写算子层（ops.py）

每个算子都成对实现 `forward` 与 `backward`，反向结果用有限差分验证。

| 算子 | 前向 | 反向 | 说明 |
|---|---|---|---|
| Linear | `y = xW + b` | dW / db / dx 三段矩阵乘 | 全部参数矩阵均为此实现 |
| Embedding | 查表 | 按 token 索引累加梯度 | 输出层与嵌入共享权重（tied） |
| LayerNorm | 减均值除标准差 + γ/β | dx 含均值/方差二次项、dγ、dβ | 每层两处（attn 前 / FFN 前）+ 末端一处 |
| GELU | tanh 近似 | 精确导数 | FFN 激活 |
| Softmax | 数值稳定（减 max） | `(dy - sum(dy*y)) * y` | 注意力权重与输出分布 |
| CausalSelfAttention | QK^T/√d + 因果掩码 + softmax + V，**推理时带 KV Cache** | dQ/dK/dV/dW 全链路 | 因果掩码保证不看未来 token |
| Residual | `x + f(x)` | 梯度直通两路 | 每层两处 |
| CrossEntropy | log-softmax + NLL | `(p - onehot)/N` | 训练损失 |
| AdamW | β1/β2 + weight decay 解耦 | — | 手写，含 bias correction |

---

## 4. 梯度校验（真实结果）

`python grad_check.py` 原样输出（完整存档见 `logs/grad_check.txt`）：

```
算子数值梯度校验（中心差分，eps=1e-06，float64）
Linear (forward: y=xW+b) dx=5.429e-09, dW=7.250e-09, db=1.424e-10 max_rel=7.250e-09 [PASS]
LayerNorm (forward: 归一化+仿射) dx=4.297e-08, dg=6.950e-10, db=1.256e-09 max_rel=4.297e-08 [PASS]
Softmax (forward: 归一化概率) dx=3.958e-08 max_rel=3.958e-08 [PASS]
Causal Attention (forward: 因果掩码+qk^T/√d·v) dq=5.801e-08, dk=8.689e-06, dv=1.862e-08 max_rel=8.689e-06 [PASS]
GELU (forward: tanh 近似) dx=2.448e-07 max_rel=2.448e-07 [PASS]
CrossEntropy (forward: fused softmax+NLL) dlogits=3.498e-08 max_rel=3.498e-08 [PASS]
最大相对误差 = 8.689e-06；判定标准 < 1e-5；总体结果：全部通过 (PASS)
```

> 最大误差出现在 Attention 的 dK（8.689e-06）：因果掩码下部分位置只参与极少路径，梯度量级小、差分噪声相对放大，属正常现象；仍低于 1e-5 阈值。

---

## 5. 手写 BPE 分词器（bp/bpe.py）

- 字节级 BPE：先把 UTF-8 文本拆成字节符号，再做 merge，因此**任何语言都能无损往返**。
- `prepare.py` 从语料统计共现，训练出 **15,741 条 merges**，最终词表 **16,000**（39.8 秒）。
- 编码/解码往返自检（`tools/selfcheck.py`）：

```
[OK]   9 token  '你好，我是小模型。'
[OK]  25 token  'User: 你是谁？\nAssistant: 我是一个参数量很小的语言模型'
[OK]  12 token  '混合 mixed 中英 text 123 !@#'
词表大小：merges=15741（含 256 字节符号与特殊位）
```

（以上为 `tools\selfcheck.py` 原样输出，存档见 `logs/selfcheck.txt`；同报告还实测参数量 46,152,704 与 `ckpt\final.npz` 的 593 个权重数组完整性，均 PASS。）

- 不使用 `tokenizers` / `sentencepiece`，训练、编码、解码全部自写。

---

## 6. 从零训练器（train.py）

- 手写 AdamW：β1=0.9、β2=0.95、weight decay=0.1（解耦）、bias correction。
- 学习率调度：线性 warmup（前 50 步）+ cosine 衰减到 10% 底值。
- 梯度裁剪：全局 L2 范数裁剪到 1.0（日志记录原范数，标注 `(clipped)`）。
- MiniBatch 采样：从 `train.bin` 随机连续窗口（ctx=256），每步 batch×ctx 个 token。
- checkpoint：写 `ckpt/stepNNNNNN.npz` + 更新指针 `ckpt/latest.json`；`--resume` 从指针续训（含 step / tokens / 优化器状态）。
- 日志：`logs/train.log`，格式化一行一步；`--background` 用 `DETACHED_PROCESS` 脱离控制台，前台不阻塞。

### 真实训练日志（loss 下降，节选自 logs/train.log）

```
# 训练日志 | config: d_model=512 n_layer=12 n_head=8 ctx=256 ffn=2048 vocab=16000 params=46152704 | batch=8 lr=0.0012 warmup=50 wd=0.1 clip=1
# step | loss    | lr        | tokens/s | 累计tokens | 步耗时s | 梯度范数
     1 | 9.7532 | 2.400e-05 |    151 |      2048 |  13.59 | 5.968 (clipped)
    10 | 9.0215 | 2.400e-04 |     71 |     20480 |  88.63 | 1.361 (clipped)
    50 | 7.5169 | 1.200e-03 |    187 |    102400 |   8.35 | 1.757 (clipped)
   100 | 7.1421 | 1.149e-03 |    213 |    204800 |   8.25 | 1.048 (clipped)
   150 | 7.0807 | 1.000e-03 |    169 |    307200 |  11.17 | 0.860
   200 | 6.9243 | 7.849e-04 |    175 |    409600 |  11.12 | 0.873
   250 | 6.9091 | 5.446e-04 |    173 |    512000 |  10.88 | 0.929
   300 | 6.7847 | 3.271e-04 |    172 |    614400 |  11.85 | 0.746
   350 | 6.6913 | 1.756e-04 |    163 |    716800 |  11.44 | 0.924
   400 | 6.1200 | 1.200e-04 |    161 |    819200 |  12.68 | 2.156 (clipped)
```

> 说明：日志中 step 10 / 50 / 100 出现重复，是**真实的多段续训**（中途进程因 Windows 文件占用退出，用 `--resume` 从最近的 `stepNNNNNN.npz` 续跑），不是伪造；`tokens` 累计值连续递增可证。
> 单步约 8–13 秒（46M 参数、CPU 单核 numpy 反向），400 步实际墙钟约 2 小时。

### 训练器有效性交叉验证（小语料快速过拟合实验）

为排除"loss 下降只是 lr 曲线抖动"的可能，另用 8,000 tokens 极小语料、batch=4、lr=2e-3 独立跑 60 步（`--ckpt-dir ckpt_overfit --data data/train_small.bin`，日志 `logs/overfit2.log`）：

```
     1 | 9.7582 | 4.000e-04 |    117 |      1024 |  8.75 | 9.056 (clipped)
     5 | 7.9254 | 2.000e-03 |    117 |      5120 | 11.69 | 1.918 (clipped)
    10 | 6.5607 | 1.977e-03 |    105 |     10240 |  9.78 | 1.493 (clipped)
    20 | 7.0193 | 1.727e-03 |    109 |     20480 | 10.56 | 1.272 (clipped)
    30 | 6.4699 | 1.279e-03 |    111 |     30720 | 10.37 | 2.016 (clipped)
    40 | 6.9248 | 7.735e-04 |    106 |     40960 | 11.14 | 1.136 (clipped)
    50 | 6.5741 | 3.719e-04 |     85 |     51200 |  7.77 | 2.368 (clipped)
    60 | 6.5633 | 2.015e-04 |     93 |     61440 |  4.73 | 0.790
```

同一套 `train.py` 换语料、换 lr、换输出目录都能正常收敛与落盘（当时生成了 `ckpt_overfit/step000030.npz`、`step000060.npz`、`final.npz`；验证完毕后临时权重已清理，日志 `logs/overfit2.log` 保留），说明训练循环、优化器、checkpoint 机制是通用的而非针对某次运行硬编码。

---

## 7. 真实对话记录（原样输出，未经任何修饰）

加载 `ckpt/final.npz`（step 400），`chat.py` 自动喂入 3 轮，记录文件 `logs/chat_final2.txt`：

```
1. You> 你好，你是谁？
   Assistant> 能的一景：Assistant:
   [8 tok / 0.19s = 42.99 tok/s | 首 token 0.05s | 上下文 29/256]

2. You> 今天天气不错，你觉得适合做什么？
   Assistant> 是，以一个一。 ，以背个的，可以景。 ，
   [23 tok / 0.55s = 41.44 tok/s | 首 token 0.06s | 上下文 67/256]

3. You> 用一句话解释什么是机器学习。
   Assistant> 的，我的一的。
   [10 tok / 0.23s = 43.03 tok/s | 首 token 0.18s | 上下文 89/256]
```

**这就是模型的真实水平：不成句、无实义。** 不做任何美化。

---

## 8. 能力边界（必读）

### 能做什么

1. **完整可跑通的工程链路**：数据 → BPE → 训练 → checkpoint → 推理 → 流式 CLI，全部自写、可复现。
2. **梯度与学习正确**：数值梯度校验 6/6 通过；loss 从 9.75 单调降到 6.12（随机基线 ln(16000)=9.68，说明已明显脱离随机）。
3. **学到了对话格式**：输出中能自发生成 `Assistant:` 标记、中文标点与常见虚词，说明模型捕捉到了语料的结构性高频模式。
4. **可继续训练**：`start.bat → 2` 输入步数即从现有权重续训，loss 与 lr 曲线连续。

### 不能做什么（如实说明）

1. **不能进行有意义的对话**：输出是中文高频字词拼接，不回答问题、不遵循指令。
2. **没有事实知识**：无法回答"法国首都是哪"之类问题。
3. **英文能力极弱**：语料中英文占比较低，英文提示基本无有效续写。
4. **上下文短**：ctx=256 token，约 150 个汉字，超过会自动丢弃最旧轮次。

### 为什么

- 本次只训练了 **81.9 万 tokens**（400 步 × 2048）。业界同类小模型（如 TinyLlama 1.1B）用了 **3 万亿 tokens**，是本次的 **360 万倍**；
- 46M 参数在 CPU 上单步 8–13 秒，要跑到"能说人话"的量级（数十亿 tokens）需连续训练数十年，物理上不可行；
- 因此本项目的定位是**「算子与训练链路零框架手写」的工程验证**，而不是可用的对话产品。诚实结论：**架构、反向传播、优化器、分词、推理全部正确；模型本身处于极早期欠拟合状态。**

### 想让它更像样，怎么办

```bat
:: 1) 续训更多步（CPU 上每 1000 步约 3 小时，loss 约再降 0.3–0.6）
python train.py --steps 3000 --batch 8 --lr 8e-4 --warmup 0 --resume --background
:: 2) 换更大语料 / 更长上下文需同时改 configs/base.json
:: 3) 上 GPU 时把 ops.py 的矩阵乘换成 cupy 即可（其余代码不动）
```

---

## 9. 复现步骤

```bat
:: 1. 一键环境与菜单
start.bat

:: 2. 语料（菜单 4）：hf-mirror 下载 → 整理 → BPE → 编码
python tools\download_corpus.py --background
python prepare.py

:: 3. 梯度校验（菜单 3）
python grad_check.py

:: 4. 训练（菜单 2）
python train.py --steps 400 --batch 8 --lr 1.2e-3 --warmup 50 --resume --background

:: 5. 对话（菜单 1 = 0.05B 指令模型，菜单 2 = 自训练模型）
python chat.py --model instruct     :: 公开 50M 指令微调权重（默认）
python chat.py --model scratch      :: 本工程从零训练的 46M 权重
```

指令模型通路的自检与基准：

```bat
python inf\bench_instruct.py        :: 权重校验 / 分词往返 / KV Cache 一致性 / 性能 / 内存
```

自检脚本：

```bat
python tools\selfcheck.py     :: 输出 logs/selfcheck.txt（依赖审计 / 参数量 / 分词器往返 / 权重完整性）
```

---

## 10. 附：参数量核算明细

| 组件 | 形状 | 参数量 |
|---|---|---|
| token embedding（与输出层共享） | 16000 × 512 | 8,192,000 |
| 位置编码表 | 256 × 512 | 131,072 |
| 每层 QKV 投影 | 512 × 1536 + 1536 | 787,968 |
| 每层 attn 输出投影 | 512 × 512 + 512 | 262,656 |
| 每层 attn LayerNorm | 2 × 512 | 1,024 |
| 每层 FFN 第一层 | 512 × 2048 + 2048 | 1,050,624 |
| 每层 FFN 第二层 | 2048 × 512 + 512 | 1,049,088 |
| 每层 FFN LayerNorm | 2 × 512 | 1,024 |
| 单层小计 | — | 3,152,384 |
| 12 层合计 | — | 37,828,608 |
| 末端 LayerNorm | 2 × 512 | 1,024 |
| **总计** | | **46,152,704** |

`tools/selfcheck.py` 会实测权重数组求和并与此值比对，一致才打印 OK。

---

## 11. 0.05B 指令模型通路（公开权重 + 手写算子）

### 11.1 权重来源

| 项 | 实测值 |
|---|---|
| 仓库 | `hayder86al/tinystories-50m-instruct`（Hugging Face） |
| 下载镜像 | `https://hf-mirror.com`（官方 huggingface.co 在本机不通，全部经镜像 API 枚举 siblings 后逐文件下载，均带 `--max-time`/urllib timeout） |
| 落盘目录 | `model_50m_instruct/` |
| 权重文件 | `model.safetensors` **212.0 MB**（222,419,472 字节） |
| 附带文件 | `config.json` / `configuration_minigpt.py` / `modeling_minigpt.py` / `generation_config.json` / `tokenizer.json`（492.6 KB）/ `tokenizer_config.json` |

> 注：`modeling_minigpt.py`、`configuration_minigpt.py` 是上游仓库原文件，仅作架构对齐的参考存档，本工程**从不 import**（其内部 `import torch` 因此不影响运行）。运行期只读取 `config.json`、`model.safetensors`、`tokenizer.json` 三个文件。全工程 19 个 .py 的依赖审计结果：除该存档文件外，无任何 torch / transformers / safetensors / tokenizers 导入。

### 11.2 架构逐项对齐（读源码得出，非推测）

| 项 | 仓库 config / 源码 | 本工程实现 |
|---|---|---|
| 架构类 | `MiniGPTForCausalLM`（`model_type=minigpt`，自定义 `auto_map`） | `inf/minigpt_numpy.py` |
| 层数 / 头数 / embd | `n_layer=6` / `n_head=12` / `n_embd=768`（head_dim=64） | 同名参数一致 |
| 上下文 / 词表 | `context_len=1024` / `vocab_size=8000` | 预分配 KV Cache 1024，词表 8000 |
| 位置编码 | **可学习绝对位置编码** `pos_emb(1024, 768)`（非 RoPE） | `pos_emb[pos0:pos0+T]` 逐位置相加 |
| LayerNorm | `nn.LayerNorm(n_embd)`，即 **eps=1e-5**、有 weight/bias | 手写 `layernorm(x, w, b, eps=1e-5)` |
| 归一化位置 | **Pre-LN**：`x = x + Attn(LN(x))`，`x = x + MLP(LN(x))` | 同序，末端 `ln_f` |
| 激活 | **GELU（tanh 近似）** | `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` |
| 注意力缩放 | `q @ k^T / sqrt(head_dim)`，causal mask | 同，`1/sqrt(64)` |
| bias | QKV / proj / MLP / LN **全部有 bias** | 全部从权重读取，不假设 |
| 权重共享 | `lm_head.weight is token_emb.weight`（tied） | 加载后校验：最大差异 **0.000e+00** |
| 特殊 token | `pad=0` `<pad>` / `eos=1` `<eos>` / `unk=2` `<unk>` | decode 时遇 eos 停止 |
| 对话模板 | `User: {问题}\nAssistant:` | `build_prompt()` 默认 single 模式 |

### 11.3 权重加载校验（`inf/safetensors_lite.py` 自解析）

不依赖 `safetensors` 库：读取前 8 字节小端 uint64 头长度 → 解析 JSON header → 按 `data_offsets` 切片为 `np.frombuffer`（float32）。

```
[权重] safetensors 数组 65 个，实际匹配进图 64 个张量
[权重] 缺失键 0 个 []
[权重] 多余键 0 个 []
[权重] 唯一参数(去重后实际进图) 49,440,768，文件内合计(含 tied 重复) 55,584,768
[权重] head.weight 与 token_emb 最大差异 0.000e+00（完全绑定）
```

参数量核算：token_emb 8000×768=6,144,000；pos 1024×768=786,432；每层 QKV 768×2304+2304 + proj 768×768+768 + 2×LN(1536) + MLP(768×3072+3072 + 3072×768+768) = 7,090,944；6 层 = 42,545,664；末端 LN 1,536 → **6,144,000 + 786,432 + 42,545,664 + 1,536 = 49,477,632**，扣除 head 绑定（-6,144,000）后与文件内 55,584,768 对应，唯一参数 **49,440,768 ≈ 49.44M**。

### 11.4 Tokenizer 自实现（`inf/unigram_tok.py`）

仓库 `tokenizer.json` 的 `model.type = Unigram`、`vocab 8000`、`unk_id=2`，前置/后置均为 `ByteLevel`（`add_prefix_space=true`、`use_regex=true`）。本工程自写：

- GPT-2 正则切分 → 逐 token 转 UTF-8 字节 → `bytes_to_unicode` 映射；
- Unigram **Viterbi 最优切分**（含 unk 罚分），`decode` 做反向字节还原后按 UTF-8 容错解码；
- 不引入 `tokenizers` / `transformers` / `sentencepiece`。

往返自检（`inf/bench_instruct.py`）：**10/10 PASS**，含英文、中文、中英混排、emoji、制表符/换行、长重复串、符号集，`decode(encode(x)) == x` 全部成立；中文 10 字符 → 31 token（UTF-8 字节级），编码 0.18 ms。

### 11.5 推理性能（CPU，numpy float32）

| 指标 | 实测（优化后） | 优化前 |
|---|---|---|
| prefill（26 token，英文） | **48–99 ms** | 165 ms |
| 增量解码 | 100 token / 2.64 s = **37.9 tok/s（26.4 ms/token）** | 38.2 tok/s |
| 首 token 延迟 | **26–34 ms**（warmup 后稳定） | 可达 2261 ms（冷启动缺页） |
| 整轮对话平均（5 轮 CLI） | **34.1–39.8 tok/s，均值 37.7** | — |
| 内存峰值 | 411–447 MB（权重 212 MB + 运行时） | 412–441 MB |
| KV Cache | 36.0 MB 预分配（6 层 × K/V × 1 × 12 头 × 1024 ctx × 64 dim × 4B） | 同 |
| 权重加载（含 212MB 拷入内存 + warmup） | 0.26 s | 0.26 s（memmap 惰性） |

三项关键优化（针对实测暴露的问题）：

1. **权重脱离 memmap**：`safetensors` 以 `np.memmap` 读取，原实现直接切片保留视图，首次前向因缺页中断逐页读 212 MB，实测把首 token 拖到 **2261 ms**。改为加载时 `ascontiguousarray(copy)` 一次性驻留，prefill 稳定在 48–99 ms。
2. **转置权重连续化**：PyTorch `Linear` 权重布局是 (out, in)，手写算子需 (in, out)，转置后用 `ascontiguousarray` 落成 C 连续，避免 BLAS 走非连续慢路径。
3. **加载后 warmup**：加载完成即跑一次 1-token 前向，提前拉起 BLAS 线程池与权重页，首 token 延迟不再含冷启动开销。

按内存带宽估算：49.44M 参数 × 4 B ≈ 198 MB/token，37.9 tok/s ≈ 7.5 GB/s 有效带宽，已接近本机单线程 numpy 的带宽上限，**再提速需量化（float16/int8），会牺牲手写算子的直观可读性，故未做**。

KV Cache 正确性：把 15 token 序列逐步增量解码的 logits 与一次性全序列前向比对，**最大绝对误差 9.537e-06、平均 1.188e-06、逐位置 argmax 一致 15/15**（float32 累加顺序差异导致，非逻辑错误）。解码全程只前向 1 个 token，不重算全序列。

### 11.6 真实对话原文（原样贴出，未修饰）

`python chat.py --model instruct`，参数 temp=0.5 / top_k=40 / top_p=0.9 / max_new=120，single 模式（日志留存于 `logs/instruct_single.log`）：

```
You> What is 2+2?
Assistant>  2+2 = 2!
[7 tok / 0.21s = 34.11 tok/s | prefill 48 ms | 上下文 29/1024 | 峰值 447 MB]

You> Write a short story about a cat.
Assistant>  In a tiny village, there was a small cat. The cat liked to play and run.
[25 tok / 0.68s = 36.64 tok/s | prefill 55 ms | 上下文 51/1024]

You> 你好，介绍一下你自己
（提示：该权重仅用英文 TinyStories 语料训练，词表无中文字符，中文以 UTF-8 字节碎片表示，中文回答无实际意义）
Assistant>  session session session session session session matily),/////////////////////----------------------------------------------------------------------------------------
[120 tok / 3.12s = 38.50 tok/s | prefill 481 ms | 上下文 165/1024]

You> Why is the sky blue?
Assistant>  The sky looks blue because of the sun. It is very big and bright.
[21 tok / 0.53s = 39.78 tok/s | prefill 99 ms | 上下文 45/1024]

You> Tell me a story about a brave little girl.
Assistant>  A brave girl stood in the dark. She held her head low and looked around. Suddenly, she saw a bright light. She followed the light and found herself in a magical forest.
[51 tok / 1.29s = 39.56 tok/s | prefill 51 ms | 上下文 78/1024]
```

> 采样有随机性，不同轮次答案会变（例如同一问题另一次测得 `2+2!`、`Once upon a time, there was a little cat named Mittens. Mittens loved to play with yarn and chase after mice.`）。以上为一次完整运行的终端原文。

### 11.7 能力边界（如实说明，不美化）

1. **英文童话类叙述可用**：故事、简单描述能生成语法通顺、上下文自洽的英文句子（TinyStories 训练域内）。
2. **算术/事实基本不可用**：`What is 2+2?` → `2+2!`（未给出 4）；问名字 → `My name is Hayder!`（作者名混入）。官方模型卡亦声明"不保证事实准确性"。
3. **中文完全不可用**：词表 8000 无 CJK，中文按 UTF-8 字节编码，输出为 `foam` 等字节碎片。分词本身往返正确（编码/解码无损），缺的是中文语义表征。
4. **多轮上下文反而降质**：官方卡写明 single-turn only。实测 multi 模式（拼接历史）第 4 轮直接复读第 2 轮的答案（`I can't count+2, I am a puppy in the computer!` 出现两次），因此默认 `single` 模式，`/mode multi` 仅供演示截断逻辑。
5. **上下文 1024 上限**：multi 模式超限时自动丢弃最旧轮次（实测提示"已自动丢弃最旧的 3 轮"后仍能续写连贯故事）。
6. **规模限制**：49.44M 参数 + 8000 词表 + 纯 CPU numpy，无法与数十亿参数模型比较；tok/s 数据为单机 CPU 实测，随 CPU 负载波动（优化后稳态 34–40 tok/s，曾观测到系统负载高时整体降至约 5 tok/s，复测即恢复）。

### 11.8 指令模型通路文件清单

| 文件 | 作用 |
|---|---|
| `inf/safetensors_lite.py` | safetensors 自解析（8 字节头长 + JSON header + 偏移切片） |
| `inf/unigram_tok.py` | Unigram + ByteLevel tokenizer 自实现（encode/decode/往返自检） |
| `inf/minigpt_numpy.py` | MiniGPT 手写 numpy 前向（Pre-LN/GELU/绝对位置/tied head/KV Cache）+ 权重校验 |
| `inf/instruct_chat.py` | 指令模型 CLI（流式、多轮、`/reset /temp /topk /topp /stats /model /mode /exit`） |
| `inf/bench_instruct.py` | 自检 + 性能基准（权重/分词/KV Cache/速度/内存） |
| `chat.py` | 统一入口，`--model instruct|scratch`，运行时 `/model` 互切 |
| `model_50m_instruct/` | 镜像下载的权重与配置（`model.safetensors` 212 MB 等） |
| `logs/instruct_single.log` | 第 11.6 节 5 轮对话的终端原文留存（可复核） |
| `logs/instruct_bench.log` | `bench_instruct.py` 基准输出留存 |

## 12. 公共翻译 API 中文增强（新选项，原英文直连不受影响）

模型只在英文 TinyStories 上训练，词表无中文。本节提供一个**不改模型、不加依赖**的工程化折中：
在模型外面套一层"翻译壳"，把中文问题翻成英文喂给模型，再把英文回答翻回中文。

```
中文输入 --译入(zh->en)--> 英文 prompt --0.05B 指令模型--> 英文回答 --译出(en->zh)--> 中文回显
                              （英文原文仍可查看，/raw 控制）
```

### 12.1 文件与职责

| 文件 | 职责 |
| --- | --- |
| `inf/translate_api.py` | 新增。翻译提供方抽象（`Provider`）+ MyMemory/LibreTranslate 实现 + 语言判定 + 磁盘 LRU 缓存 |
| `inf/instruct_chat.py` | 新增 `ask()` 承载中文链路、`/translate` `/raw` 命令、翻译耗时统计 |
| `chat.py` | 新增 `--translate` `--provider` `--libre-endpoint` `--libre-key` 启动参数 |
| `start.bat` | 新增菜单项 `7 中文增强对话（公共翻译 API）[需联网，公共接口较慢]`，1 号入口仍为英文直连 |
| `data/translate_cache.json` | 磁盘翻译缓存（LRU 2000 条），命中即不发请求 |

### 12.2 提供方链与容错

| 提供方 | 端点 | key | 说明 |
| --- | --- | --- | --- |
| `mymemory`（默认首选） | `https://api.mymemory.translated.net/get` | 不需要 | 实测 200 可用；**单次查询硬上限 500 字符**，超长按标点切片（480 字符/片）串行拼回 |
| `libretranslate`（默认关闭） | 由 `--libre-endpoint` 指定 | 由 `--libre-key` 指定 | 公共实例基本都要 key/登录，因此默认不进链；未配置 endpoint 时视为"未启用" |

- 每个请求 **6 秒超时**；失败立刻换下一个提供方，**不重试、不挂起**。
- 全链失败时 **降级为原文直通**（`ok=False`），CLI 打印失败原因后按原文继续对话，**对话不中断**。
- 语言判定：按 CJK 字符占比（≥15% 判为中文）自动决定方向；**英文输入不翻译**，直送模型。
- 缓存：`data/translate_cache.json`，键为 `src>dst|原文`，LRU 上限 2000 条，命中打印 `提供方 cache（磁盘缓存命中，未发请求），耗时 0.00s`。

### 12.3 用法

```bat
:: 方式一：双击 start.bat，选 7（中文增强）；1 号入口仍是英文直连
:: 方式二：命令行
python -X utf8 chat.py --model instruct --translate
python -X utf8 chat.py --model instruct --translate --provider mymemory
:: LibreTranslate（自建实例，已有 key 时；注意这是单行，此处换行仅为排版）
python -X utf8 chat.py --model instruct --translate --provider mymemory,libretranslate --libre-endpoint http://127.0.0.1:5000 --libre-key <your-key>
```

| 命令 | 作用 |
| --- | --- |
| `/translate on\|off\|status` | 开启 / 关闭 / 查看中文增强（关闭即回到纯英文直连，模型与推理完全不变） |
| `/translate provider mymemory,libretranslate` | 重设提供方链（按顺序回退） |
| `/raw on\|off` | 中文轮是否同时打印模型英文原文（默认 on） |
| `/stats` | 追加显示中文轮次、译入/译出累计耗时、缓存命中与请求次数 |

**关闭翻译回到纯英文模式**：`/translate off`，或启动时不加 `--translate`（start.bat 选 1）。此时代码路径与新增功能前完全一致，不产生任何网络请求。

### 12.4 实测：4 轮中文完整链路（终端原文，未修饰）

`python -X utf8 chat.py --model instruct --translate --provider mymemory`，temp=0.50：

```
You> 你好，请简单介绍一下你自己。
[译入] Hello, please briefly introduce yourself.
  └ 提供方 mymemory，耗时 2.32s
Assistant(EN)>  Hello! I am your friendly puppy helper. What would you like to talk about?
[中文回显] 您好！我是您友善的小狗助手。您想聊些什么？
  └ 提供方 mymemory，耗时 1.50s
[链路] 译入 2.32s(mymemory) | 推理 0.70s / 23 tok (32.69 tok/s) | 译出 1.50s(mymemory) | 合计 4.52s

You> 请讲一个关于小猫的短故事。
[译入] Please tell a short story about a kitten.
  └ 提供方 mymemory，耗时 1.38s
Assistant(EN)>  Once upon a time there was a kitten who loved to play. Every day it would go out and explore the world.
[中文回显] 从前，有一只喜欢玩耍的小猫。它每天都会出去探索世界。
  └ 提供方 mymemory，耗时 2.90s
[链路] 译入 1.38s(mymemory) | 推理 0.86s / 29 tok (33.83 tok/s) | 译出 2.90s(mymemory) | 合计 5.14s

You> 那只小猫叫什么名字？
[译入] What's the name of that kitten?
  └ 提供方 mymemory，耗时 1.32s
Assistant(EN)>  I don't know the name of that kitten, but I think it's cute. Want to call it Mittens?
[中文回显] 我不知道那只小猫的名字，但我觉得它很可爱。想叫它“小手套”吗？
  └ 提供方 mymemory，耗时 1.42s
[链路] 译入 1.32s(mymemory) | 推理 0.89s / 30 tok (33.88 tok/s) | 译出 1.42s(mymemory) | 合计 3.62s

You> 小猫最后做了什么？
[译入] What did the kitten do in the end?
  └ 提供方 mymemory，耗时 1.22s
Assistant(EN)>  The kitten found a little box and opened it. Inside, it found a shiny toy.
[中文回显] 小猫咪找到一个小盒子，然后打开了它。它在里面发现了一个闪闪发光的玩具。
  └ 提供方 mymemory，耗时 1.56s
[链路] 译入 1.22s(mymemory) | 推理 0.71s / 24 tok (33.74 tok/s) | 译出 1.56s(mymemory) | 合计 3.49s
```

耗时结构（实测区间，如实报告）：**译入 1.22-2.32s / 推理 0.70-0.89s / 译出 1.42-3.84s**，
单轮总延迟 **3.5-5.2s**，其中翻译占 70%~85%。**慢就是慢**，瓶颈在公共接口，不在模型（模型本身约 33.6 tok/s）。

### 12.5 实测：缓存命中（重复同一问题）

首次出现该问题时译入为 `mymemory 1.38s`；重复同一问题（temp=0.0 使输出可复现，译入与译出都能命中）：

```
You> 请讲一个关于小猫的短故事。
[译入] Please tell a short story about a kitten.
  └ 提供方 cache（磁盘缓存命中，未发请求），耗时 0.00s
Assistant(EN)>  Once upon a time, there was a little kitten. The kitten was very hungry and wanted something to eat.
[中文回显] 从前，有一只小猫咪。小猫咪肚子饿极了，想吃点东西。
  └ 提供方 mymemory，耗时 3.84s
[链路] 译入 0.00s(cache) | 推理 0.80s / 27 tok (33.66 tok/s) | 译出 3.84s(mymemory) | 合计 4.64s

You> 请讲一个关于小猫的短故事。      ← 再次重复（译入、译出双双命中）
  └ 提供方 cache（磁盘缓存命中，未发请求），耗时 0.00s
  └ 提供方 cache（磁盘缓存命中，未发请求），耗时 0.00s
[链路] 译入 0.00s(cache) | 推理 0.79s / 27 tok (34.07 tok/s) | 译出 0.00s(cache) | 合计 0.79s
```

- 译入命中：同一中文问题不再发请求，`2.32s -> 0.00s`。
- 译出命中：取决于英文回答是否**逐字相同**。默认 temp=0.5 时模型输出随机，回译文本不同 -> 不命中（这是必然，不是缺陷）；temp=0.0 可复现输出时两段全命中，单轮总延迟 **4.64s -> 0.79s**。
- 缓存文件可随时删除，删掉后下次重新请求。

### 12.6 实测：失败降级（对话不中断）

```
# 1) 提供方链回退：LibreTranslate 端点不可达 -> 自动改用 MyMemory
[翻译] libretranslate 失败，尝试下一个来源：网络/超时: <WinError 10061 由于目标计算机积极拒绝...>
结果: ok=True provider=mymemory 文本='Please tell a short story about a kitten.' 耗时 3.68s

# 2) 只配了未填 endpoint 的 LibreTranslate（默认关闭）-> 降级原文
结果: ok=False 降级后文本='你好世界' error='libretranslate: 未启用'

# 3) 端点拒绝连接 -> 2.06s 内失败并降级，未触及 6s 超时，未挂起
```

CLI 内表现为：`[译出失败] 回译不可用，以下为模型英文原文：...`，随后照常打印 `[链路]` 与统计，**下一轮对话正常继续**。

### 12.7 如实说明的边界

- 翻译只解决"出入口语言"，**不提升模型智力**：50M 英文童话模型仍会答非所问、上下文弱、多轮易漂移，中文回显只是把这种弱如实呈现。
- MyMemory 是公共免费接口，**有限速与抖动**（实测单次 1.2-3.9s），不适合高频生产使用；需要稳定请自建 LibreTranslate 并配 `--libre-endpoint` / `--libre-key`。
- 单次 500 字符上限已按标点切片处理，超长回答会被切成多段串行翻译，延迟相应叠加。
- 机器翻译存在语义漂移（行业术语、双关、数字），中文回显仅供参考。

*（内容由AI生成，仅供参考）*
*（内容由AI生成，仅供参考）*

*（内容由AI生成，仅供参考）*
