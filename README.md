# llm005b

**从零手写算子训练的 46M 参数语言模型，纯 NumPy 实现，不依赖 PyTorch / JAX / TensorFlow。**

前向、反向传播、AdamW、BPE 分词器、推理引擎全部手写，在一台普通 Windows 笔记本的 CPU 上完成训练与推理。

- 参数量：**46,152,704**（46.15M）
- 训练进度：**400 步 / 约 82 万 tokens**，loss **6.12**
- 依赖：**只有 numpy**
- 训练与推理硬件：纯 CPU（8GB 内存可跑）

> 这是「AI 本体」的开源：源码、分词器、训练出来的权重、构建与自检脚本一并公开。

---

## 这不是什么

请先看清楚边界，再决定要不要用：

- **不是可用的对话产品**。400 步、82 万 tokens 的训练量只够一个工程基线，模型输出以碎片化文本为主，无法正常聊天、没有事实知识、没有算术能力。
- **不是为了刷指标**。它的价值在于：所有算子、梯度、优化器、分词器、推理路径都是可读、可改、可验证的明文实现，适合作为学习与二次开发的起点。
- 想要「能对话」的体验，仓库里另有 `inf/instruct_chat.py`，用于加载外部 50M 指令模型（见下文「两条推理线」），与自训权重不是同一个模型。

---

## 目录结构

| 路径 | 作用 |
| --- | --- |
| `ops.py` | 手写算子：Linear / LayerNorm / 注意力 / GELU 的前向与反向 |
| `model.py` | GPT 组装（wte + wpe + 12 层 + final LN），`num_params()` 参数口径固定 |
| `train.py` | AdamW + warmup + 余弦退火 + 梯度裁剪，支持断点续训 |
| `grad_check.py` | 数值梯度校验：逐算子对比解析梯度 |
| `bp/bpe.py` | 自训 byte-level BPE 分词器 |
| `bp/tokenizer.json` | 训练好的分词器（merges 15741，vocab 16000） |
| `prepare.py` | 语料 → 二值 token 流，并按 ctx 切分为训练样本 |
| `inf/minigpt_numpy.py` | 纯 numpy 推理引擎，可读 `.npz` 与 safetensors |
| `inf/instruct_chat.py` | 指令对话 CLI（加载外部 50M 指令模型） |
| `inf/safetensors_lite.py` | 纯 Python safetensors 读取实现 |
| `inf/translate_api.py` | 可选的中英桥接（MyMemory 公共 API，仅用于中文问句输入） |
| `inf/bench_instruct.py` | 推理速度基准 |
| `tools/download_corpus.py` | 对话语料下载整理（纯标准库） |
| `tools/selfcheck.py` | 依赖审计 / 参数量 / 分词往返 / 权重完整性自检 |
| `tools/mem_probe.py` | 内存占用探测 |
| `chat.py`、`start.bat` | 一键入口与菜单 |
| `configs/base.json` | 模型配置 |

---

## 模型规格

| 项 | 值 |
| --- | --- |
| 参数量 | 46,152,704 |
| 层数 `n_layer` | 12 |
| 隐藏维度 `d_model` | 512 |
| 注意力头数 `n_head` | 8 |
| FFN 维度 | 2048 |
| 上下文长度 `ctx` | 256 |
| 词表大小 `vocab_size` | 16000 |
| 词嵌入 | 输入输出权重共享（`tied: true`） |
| 位置编码 | 可学习 wpe（256 × 512） |
| 归一化 | LayerNorm（Pre-LN） |
| 激活 | GELU |
| 优化器 | AdamW，warmup + 余弦退火，梯度裁剪 |

---

## 快速开始

### 1. 环境

```bash
pip install numpy
```

只有这一个依赖。`tools/selfcheck.py` 会做依赖审计，确认全文没有 `torch` / `transformers` / `jax` / `onnx` / `tensorflow` 引用。

### 2. 自检

```bash
python tools/selfcheck.py
```

输出依赖审计、参数量明细、分词器往返、权重完整性四段报告。

### 3. 梯度校验

```bash
python grad_check.py
```

用中心差分对每个手写算子做数值梯度比对，解析梯度与数值梯度不匹配会直接报错。

### 4. 准备语料并训练

```bash
python tools/download_corpus.py     # 下载整理对话语料到 data/
python prepare.py                   # 切成训练样本
python train.py --steps 400         # 训练，产出 ckpt/final.npz
```

### 5. 推理

```bash
python inf/minigpt_numpy.py --ckpt ckpt/final.npz --prompt "User: hi\nAssistant:"
```

---

## 实测数据

以下数据来自本机实跑日志，不是估算：

| 指标 | 实测值 |
| --- | --- |
| 参数量 | 46,152,704（自检与 `num_params()` 一致） |
| 训练步数 | 400 |
| 训练 tokens | 约 81.9 万 |
| 最终 loss | 6.1200 |
| 权重文件 | `ckpt/final.npz`，554.0 MB |
| 分词器 | merges 15741，含 256 字节符号与特殊位 |
| 分词器往返 | 中英混排样本全部往返一致 |
| 依赖审计 | 通过（仅 numpy） |
| 推理速度（外部 50M 指令模型） | 37.9 tok/s，纯 numpy，单核 CPU |

---

## 权重

`ckpt/final.npz`（554 MB）超过 git 仓库的舒适体积，因此作为 **Release 附件**分发，不打进 git 历史：

- 下载：见本仓库 `Releases` 页面（标签 `v1.0.0`，资产 `final.npz`）
- 放置到仓库的 `ckpt/final.npz` 后即可直接推理

若只需要「能对话」的体验而非自训基线，`inf/instruct_chat.py` 会加载外部指令模型（`hayder86al/tinystories-50m-instruct`，MIT），请自行从其官方页面获取权重，本仓库不随附。

---

## 训练数据

**本仓库不分发任何原始语料。** `tools/download_corpus.py` 只提供下载与整理脚本，语料由使用者自行获取，并自行遵守各上游数据集的许可协议。

脚本按顺序尝试以下公开数据集（经 `hf-mirror.com` 镜像）：

| 数据集 | 上游许可 |
| --- | --- |
| `databricks/databricks-dolly-15k` | CC BY-SA 3.0 |
| `tatsu-lab/alpaca` | Apache-2.0 |
| `shibing624/alpaca-zh` | Apache-2.0 |
| `shibing624/sharegpt_gpt4` | 见上游（含使用限制，请自行确认） |

Release 中的自训权重是在上述语料上训练得到的，使用者需同时评估上游数据许可对下游用途的约束。详见 [NOTICE](NOTICE)。

---

## 两条推理线

仓库里有两套可跑的推理路径，请不要混淆：

1. **自训 46M 模型**（本仓库的「AI 本体」）
   - 规格：vocab 16000 / ctx 256 / 12 层 / 8 头 / d_model 512
   - 权重：Release 附件 `final.npz`
   - 现状：训练量极小，输出以碎片化文本为主
2. **外部 50M 指令模型**（用于验证推理引擎正确性）
   - 规格：vocab 8000 / ctx 1024 / 6 层 / 12 头 / d_model 768，49.44M 参数
   - 权重：自行从其官方页面获取，不随本仓库分发
   - 现状：英文童话式叙述通顺，可用；无中文能力（词表无 CJK），无算术能力

`inf/translate_api.py` 只是在第二套前面加了一层可选中英桥接，让中文问句能进入英文模型，不改变模型本身能力。

---

## 许可

- 代码与自训权重：**MIT**，见 [LICENSE](LICENSE)
- 第三方数据集与外部模型：各自许可，见 [NOTICE](NOTICE)

---

<sub>本站内容由 AI 生成，仅供参考。</sub>
