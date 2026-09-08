<p align="center">
  <img src="xpomni_logo.png" alt="XPeng Omni Team" width="320"/>
</p>

<h1 align="center">X-AuT</h1>

<p align="center">
  <b>基于跨尺度蒸馏的语音大模型渐进式音频编码器压缩</b><br/>
  XPeng Inc. &nbsp;|&nbsp; 🌐 <a href="https://x-aut.github.io/">项目主页</a> &nbsp;|&nbsp; 🤗 <a href="https://huggingface.co/X-AuT/X-AuT">模型权重</a> &nbsp;|&nbsp; <a href="README.md">【English README】</a>
</p>

---

## 📖 简介

**X-AuT-14layer** 是 [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) 的压缩变体，将音频塔从 18 个 Transformer 块压缩至 14 个，参数量由 **186.376M 降至 147.794M（−20.70%）**。预训练语言模型主干保持冻结；恢复训练通过 LoRA 调整解码器注意力，并在蒸馏阶段训练共享的输出嵌入，在最终微调阶段将其冻结。

移除编码器层会改变解码器接收的音频嵌入，并可能引发过早输出结束符和严重删除错误。X-AuT 将这种失配作为关键恢复目标，同时承认保留的编码器容量同样会影响结果。框架包含以下部分：

- **行为驱动的层筛选（Behavior-driven layer screening）**：在固定开发/验证子集上，通过相同短预算恢复实验比较候选配置。模型先移除原始层 {1, 18}，完成 18 → 16；再移除筛选出的 {5, 6}，完成 16 → 14。实验表明所测层对存在非加性相互作用，因此需要显式评估组合，但这并不意味着相邻层在所有模型中都更适合移除。
- **转录一致性筛选（Transcript-consistency filtering）**：对超过 28 万小时的源数据池，基于原始转录和两个离线 ASR 假设划分九个置信等级。论文中的三个训练阶段均使用 **class 1**；Stage 2 仅调整数据源权重以偏向目标领域。因此，当前结果不用于证明多等级课程学习的效果。
- **三阶段恢复**：Stage 0 对齐中间层、桥接表示与 logits；Stage 1 将 teacher-forced 蒸馏与按计划采样的学生策略上下文结合；Stage 2 使用金标转录进行较低学习率的 LoRA 微调。
- **跨尺度蒸馏（Cross-scale distillation）**：冻结的 [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 教师模型通过可学习的 2048→1024 投影监督学生，并在推理时移除。在匹配的 16 层 Stage 1 训练方案下，跨尺度教师的宏平均错误率为 5.55%，同尺度自蒸馏为 8.45%。该结果来自单次实验，是描述性比较，不能视为教师模型“创造新能力”的因果证明。

在十个公开中英基准上，16 层 Stage 2 模型将宏平均错误率从 **5.61% 降至 5.27%（相对下降 6.1%）**。14 层 Stage 2 模型的宏平均错误率为 **5.75%（相对上升 2.5%）**，同时减少 20.70% 的音频塔参数。14 层模型的编码器延迟在车载 PPU 和 H800 上分别降低 **21.4%** 和 **11.4%**。

## 🏗️ 模型架构

仅移除音频塔中的 Transformer 块；ConvStem、Bridge 和 `ln_post` 的结构保持不变。语言模型基础权重不参与剪枝，但恢复训练会在其注意力投影上添加 LoRA 适配器。

### AuT（音频编码器）配置

| 项目 | 数值 |
|------|-------|
| model_type | `qwen3_asr_audio_encoder` |
| d_model | 896 |
| Attention heads | 14 |
| FFN dim | 3584 |
| Mel bins | 128 |
| Output dim（bridge → LLM hidden） | 1024 |
| 编码器层数（原始 / 本模型） | 18 / **14** |
| 保留的原始层（从 1 开始计） | `[2,3,4,7,8,9,10,11,12,13,14,15,16,17]`（丢弃 {1,18}，再丢弃 {5,6}） |

### 参数量明细

| 模块 | AuT-18（基线） | AuT-14（本模型） |
|--------|------------------:|--------------------:|
| ConvStem（conv2d1/2/3 + conv_out） | 11.03M | 11.03M |
| Transformer 编码器层 | 173.62M（18 层） | **135.04M（14 层）** |
| Bridge（proj1 + proj2） | 1.72M | 1.72M |
| ln_post | 0.002M | 0.002M |
| **音频塔总计** | **≈186.38M** | **≈147.79M（↓20.7%）** |

### 全模型参数量（供参考）

| 模块 | 参数量 | 说明 |
|--------|-------:|------|
| 音频塔（AuT） | 147.794M | 已剪枝（186.376M → 147.794M） |
| 文本解码器（28 层） | 440.47M | Qwen3-0.6B，冻结 |
| Token 嵌入 | 155.58M | 与 `lm_head` 共享 |
| **总计（去重共享权重后）** | **≈743.84M** | — |

## 📊 评测

下表报告十个公开中英基准上的错误率（中文为 CER，英文为 WER），数值越低越好。论文采用十个基准等权重的宏平均作为汇总指标。所有数值均为单次运行的最佳检查点结果，加粗不表示统计显著性。

### 精度—压缩率权衡

| 模型 | 音频塔参数量 | 参数变化 | 宏平均错误率 | 相对宏平均错误率变化 |
|------|-------------:|---------:|-------------:|---------------------:|
| Full-18 基线 | 186.376M | — | 5.61 | — |
| X-AuT-16layer（Stage 2） | 167.085M | −10.35% | **5.27** | **−6.1%** |
| X-AuT-14layer（Stage 2） | 147.794M | −20.70% | 5.75 | +2.5% |

### X-AuT-14layer 详细结果

| 基准 | Full-18 基线 | X-AuT-14layer（Stage 1） | X-AuT-14layer（Stage 2） |
|-----------|:---:|:---:|:---:|
| AISHELL-1（CER） | **3.33%** | 3.52% | 3.39% |
| Fleurs-zh（CER） | **2.80%** | 3.49% | 3.32% |
| Fleurs-en（WER） | **4.17%** | 5.24% | 5.10% |
| LibriSpeech test-clean（WER） | 2.48% | 3.09% | **2.45%** |
| THCHS-30（CER） | **3.87%** | 4.23% | 4.17% |
| Tedlium（WER） | **3.35%** | 4.07% | 3.95% |
| LibriSpeech test-other（WER） | **5.39%** | 7.00% | 5.52% |
| CommonVoice v15 zh（CER） | 9.95% | 9.54% | **8.36%** |
| CommonVoice v15 en（WER） | **12.35%** | 13.94% | 12.49% |
| WenetSpeech-meeting（CER） | **8.36%** | 10.71% | 8.78% |
| *宏平均（%）* | **5.61** | 6.48 | 5.75 |
| *相对宏平均错误率变化（%）* | — | +15.5 | +2.5 |

微调后的 14 层模型在 LibriSpeech test-clean 和 CommonVoice zh 上优于基线，其余八个基准的错误率高于基线；最大退化出现在 Fleurs-en，为 +0.93 个百分点。由于尚未进行多随机种子实验和样本级置信区间分析，这些差异应视为描述性结果。

### 推理效率

| 指标 | PPU（车载） | GPU（H800） |
|--------|:---:|:---:|
| 编码器延迟 vs AuT-18 | **↓21.4%** | **↓11.4%** |
| 端到端延迟 vs AuT-18 | ↓4.7% | ↓2.6% |
| 峰值显存 vs AuT-18 | ↓4.4% | ↓2.8% |

> 端到端提升相对温和，因为未剪枝的 28 层文本解码器主导了总推理时间。测量值来自 50 条以上不同时长音频的平均结果，未保留多次运行方差。

## 📦 本仓库开源范围

本仓库公开使用 14 层检查点和进行轻量 LoRA 适配所需的组件：

| 组件 | 是否包含 | 范围 |
|------|:--------:|------|
| 独立检查点推理 | ✅ | [`infer_xaut.py`](infer_xaut.py) |
| 最简 LoRA 微调 | ✅ | [`xaut-ft-simple/`](xaut-ft-simple/) |
| 示例 manifest | ✅ | 1,000 条元数据，音频引用为占位符；不重新分发音频 |
| 行为探测与层筛选流水线 | ❌ | 不在本次代码开源范围内 |
| Stage 0/1 跨尺度蒸馏 | ❌ | 不在本次代码开源范围内 |
| 28 万小时数据、标注流水线及专有数据 | ❌ | 论文中描述，但不对外分发 |

`xaut-ft-simple/` 是一个实用的 **Stage-2 风格适配示例**，并非论文三阶段训练流程或结果表的完整复现包。

## 🚀 推理

发布的权重托管在 🤗 [X-AuT/X-AuT](https://huggingface.co/X-AuT/X-AuT)，为 **完整微调模型**，采用 safetensors 格式（音频塔已是 14 层）——可直接加载，无需下载基础模型或手动剪枝层。

我们提供了一个独立的推理脚本 [`infer_xaut.py`](infer_xaut.py)，它**不依赖** X-AuT 训练代码库——仅需 `torch`、`torchaudio`、`transformers`、`qwen-asr` 和 `huggingface_hub`（音频会自动转换为 16 kHz 单声道）。推理逻辑遵循官方流程：带语言控制后缀的 chat-template prompt → `processor` 编码 → `thinker.generate()` → 文本解码。

### 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
# 先安装与本机加速卡匹配的 PyTorch/torchaudio，再执行：
pip install -r requirements.txt
```

已验证的上层依赖版本记录在 [`requirements.txt`](requirements.txt) 中。仓库不固定 PyTorch 版本，因为 CUDA、PPU 与 CPU 环境需要不同的厂商构建。

### 快速开始

```bash
# 检查点会自动从 https://huggingface.co/X-AuT/X-AuT 下载
python infer_xaut.py \
  --audio /path/to/test.wav \
  --lang-code zh \
  --device cuda
```

或直接在 Python 中加载模型：

```python
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

processor = Qwen3ASRProcessor.from_pretrained("X-AuT/X-AuT", fix_mistral_regex=True)
model = Qwen3ASRForConditionalGeneration.from_pretrained(
    "X-AuT/X-AuT", dtype="bfloat16", device_map="cuda",
)
# 然后按照标准 Qwen3-ASR 生成流程使用
```

### 参数说明

| 参数 | 默认值 | 说明 |
|----------|---------|-------------|
| `--audio` | *(必填)* | 输入音频路径（wav/mp3/flac…），自动重采样为 16 kHz 单声道 |
| `--model` | `X-AuT/X-AuT` | Hugging Face 仓库 ID 或本地检查点目录 |
| `--lang-code` | `zh` | 语言控制前缀（`zh` / `en`） |
| `--device` | `cuda` | `cuda` 或 `cpu` |
| `--attn-implementation` | `sdpa` | `sdpa` / `flash_attention_2` / `eager` |
| `--dtype` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `--max-new-tokens` | `256` | 最大生成 token 数 |
| `--num-beams` | `1` | Beam size（默认贪心解码） |
| `--sr` | `16000` | 目标采样率 |

### 输出

每次运行输出一行 JSON：

```json
{
  "audio": "/path/to/test.wav",
  "lang_code": "zh",
  "duration_sec": 5.14,
  "raw_pred_text": "原始解码文本",
  "pred_text": "归一化后的识别文本"
}
```

- `raw_pred_text`：解码器输出的原始字符串；
- `pred_text`：由 `qwen_asr.inference.utils.parse_asr_output` 生成的归一化转录文本。

## 🎛️ 微调

我们在 [`xaut-ft-simple/`](xaut-ft-simple/) 中提供了一个最小、自包含的 LoRA 微调方案。它包含一个精简训练脚本、一份示例配置和 1,000 条元数据记录（音频引用为占位符），不依赖私有的 X-AuT 训练代码库。该示例使用转录交叉熵训练音频编码器、Bridge 和解码器 LoRA 适配器，不实现论文 Stage 0/1 的教师损失或行为驱动剪枝探测。用法、数据格式和配置细节请参见 [`xaut-ft-simple/README.md`](xaut-ft-simple/README.md)。

快速开始：

```bash
pip install -r xaut-ft-simple/requirements.txt
cd xaut-ft-simple
bash run_ft.sh
```

准备真实数据前，请回到仓库根目录，并用自动生成的无版权合成音频验证模型加载、音频预处理、前向/反向传播和一次优化器更新：

```bash
python xaut-ft-simple/scripts/smoke_train.py --device-type cuda
```

已执行的 PPU 训练路径验证及“未验证完整权重”的边界说明见 [`xaut-ft-simple/VALIDATION.md`](xaut-ft-simple/VALIDATION.md)。

## 📁 检查点

发布的检查点是 🤗 [X-AuT/X-AuT](https://huggingface.co/X-AuT/X-AuT) 上的完整模型包：完整微调权重（音频塔已是 14 层），以及 `config.json`、`generation_config.json`、tokenizer 和预处理器——推理时无需额外配置文件。许可证：[CC BY-NC 4.0](LICENSE)。

模型权重和示例数据仅供研究和评估使用。未经小鹏汽车（XPeng Inc.）事先书面许可，禁止将其用于商业用途、生产部署、转售、再许可、再分发，或用于训练/改进商业产品或服务。

## ⚠️ 可复现性说明

- 公开基准结果来自单次运行，并通过规模较小的固定开发/验证子集选择检查点，不应解读为具有统计显著性。
- 开源的 LoRA 示例不复现论文使用的私有数据配比、跨尺度教师训练、检查点选择流程或硬件测试。
- 示例 JSONL 仅包含转录文本和占位音频引用。请替换为您有权使用的音频；若修改文件位置，请同步更新 `data/train_index.json`。
- 用于模型训练的 28 万小时多系统标注流水线数据由内部授权数据、专有数据或其他合法授权数据构成。该训练数据不随本仓库发布。本次发布不包含任何第三方音频、转录文本或数据集的再分发。
- 显存占用和训练吞吐取决于本地 PyTorch、CUDA、注意力后端、音频时长与批配置。

---

## 📚 引用

如果您在研究中使用了 X-AuT，请考虑引用我们的论文：

```bibtex
@article{zhang2026xaut,
  title   = {X-AuT: Progressive Audio-Encoder Compression for
             Speech LLMs with Cross-Scale Distillation},
  author  = {Zhang, Haojun and Zou, Yi and Chen, Min and Yu, Qize and
             Fan, Lianrui and Ding, Xini and Zhou, Shuchang and
             Liu, Xianming and Huang, Shiyu},
  journal = {Preprint},
  year    = {2026},
  url     = {https://x-aut.github.io/}
}
```

---

<p align="center">
  © 2026 XPeng Inc.
</p>
