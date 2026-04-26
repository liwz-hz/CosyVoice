# CosyVoice 项目分析与使用指南

## 1. 项目简介

**CosyVoice** 是阿里巴巴 FunAudioLLM 团队开发的**基于大语言模型的文本转语音（TTS）系统**。它支持零样本多语言语音合成和声音克隆，仅需 3 秒参考音频即可克隆任意音色。

### 版本演进

| 版本 | 参数量 | 发布时间 | 核心特性 |
|------|--------|----------|----------|
| CosyVoice 1.0 | 300M | 2024-07 | 基础零样本多语言 TTS |
| CosyVoice 2.0 | 0.5B | 2024-12 | 流式合成、细粒度控制 |
| Fun-CosyVoice 3.0 | 0.5B | 2025-05 | 内容一致性、音色相似度、韵律自然度大幅提升 |

### 核心能力

- **多语言支持**：中文、英语、日语、韩语、德语、西班牙语、法语、意大利语、俄语（9 种）
- **方言支持**：广东话、闽南话、四川话、东北话、陕西话、上海话等 18+ 种中文方言
- **零样本克隆**：3 秒参考音频即可克隆音色
- **跨语言合成**：用参考音色说不同语言
- **双端流式**：文本输入 + 音频输出同时流式，延迟低至 150ms
- **指令控制**：自然语言控制情绪、语速、音量、语言等
- **发音纠正**：支持中文拼音和英文 CMU 音素发音标注

---

## 2. 系统架构

整体架构遵循 LLM-based TTS 的经典三阶段流程：

```
┌──────────┐    ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────┐
│ 文本输入  │───>│ 前端/Tokenizer│───>│ LLM 语音Token生成 │───>│ Flow Matching │───>│ HiFi-GAN  │───> 音频输出
└──────────┘    └─────────────┘    └──────────────────┘    └─────────────┘    └──────────┘
```

### 核心模块

| 模块 | 路径 | 功能 |
|------|------|------|
| Frontend | `cosyvoice/cli/frontend.py` | 文本归一化、语音 Token 提取（Whisper）、说话人特征提取（CAM++） |
| LLM | `cosyvoice/llm/llm.py` | 基于 Qwen2ForCausalLM 自回归生成语音 Token |
| Flow | `cosyvoice/flow/flow.py` | Flow Matching 将语音 Token 转换为 Mel 频谱（3.0 使用 DiT） |
| HiFi-GAN | `cosyvoice/hifigan/generator.py` | 神经声码器，Mel 频谱转音频波形 |
| Tokenizer | `cosyvoice/tokenizer/tokenizer.py` | 基于 Whisper tokenizer + tiktoken |

### 各版本 LLM 骨干网络对比

| 版本 | LLM 骨干 | 参数量 | 备注 |
|------|----------|--------|------|
| CosyVoice 1.0 | 自定义 TransformerLM | 300M | 自定义文本编码器 + 语音 LM |
| CosyVoice 2.0 | Qwen2ForCausalLM | 0.5B | HuggingFace Qwen2 |
| CosyVoice 3.0 | Qwen2ForCausalLM（增强） | 0.5B | 双端流式、混合比例训练 |

---

## 3. 技术栈

### 核心依赖

- **深度学习**：PyTorch 2.3.1 + Torchaudio 2.3.1 (CUDA 12.1)
- **LLM**：Transformers 4.51.3（HuggingFace Qwen2）
- **训练**：DeepSpeed 0.15.1、Lightning 2.2.4
- **扩散模型**：Diffusers 0.29.0
- **加速推理**：vLLM 0.9.0+、ONNX Runtime、TensorRT 10.13.3
- **服务部署**：gRPC、FastAPI、Uvicorn
- **Web UI**：Gradio 5.4.0
- **文本处理**：wetext 0.0.4（中英文文本归一化）

---

## 4. 运行示例

### 4.1 环境准备

#### 4.1.1 克隆项目

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
# 如果因网络问题子模块克隆失败，重复执行以下命令直到成功
git submodule update --init --recursive
```

#### 4.1.2 创建 Conda 环境

```bash
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt
```

#### 4.1.3 安装 sox（如需要）

```bash
# Ubuntu
sudo apt-get install sox libsox-dev
# CentOS
sudo yum install sox sox-devel
```

#### 4.1.4 系统要求

- **Python**：3.10（3.11+ 也可）
- **GPU（推荐）**：NVIDIA GPU（建议 8GB+ 显存），CUDA 12.1
- **CPU（可用但慢）**：可在 CPU 上运行，使用 `torch` CPU 版本即可，需安装 `soundfile` 保存音频

### 4.2 下载模型

推荐使用 **ModelScope SDK**（国内）或 **HuggingFace**（海外）下载：

```python
# ModelScope（国内推荐）
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')

# HuggingFace（海外推荐）
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
# ... 同上
```

### 4.3 运行 Python 示例

项目根目录的 `example.py` 包含三种版本的完整示例：

```python
# 编辑 example.py，取消注释需要运行的示例
def main():
    cosyvoice_example()   # CosyVoice 1.0 示例
    # cosyvoice2_example()  # CosyVoice 2.0 示例
    # cosyvoice3_example()  # CosyVoice 3.0 示例
```

直接运行：

```bash
python example.py
```

### 4.4 各种推理模式示例

#### 4.4.1 SFT（预训练音色合成）

```python
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT')
print(cosyvoice.list_available_spks())  # 查看可用音色

for i, output in enumerate(cosyvoice.inference_sft(
    '你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？',
    '中文女', stream=False)):
    torchaudio.save(f'sft_{i}.wav', output['tts_speech'], cosyvoice.sample_rate)
```

#### 4.4.2 零样本声音克隆

```python
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

for i, output in enumerate(cosyvoice.inference_zero_shot(
    '目标文本：收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。',
    '参考音频文本：希望你以后能够做的比我还好呦。',
    'reference_audio.wav',  # 3秒参考音频
    stream=False)):
    torchaudio.save(f'zero_shot_{i}.wav', output['tts_speech'], cosyvoice.sample_rate)
```

#### 4.4.3 跨语言声音克隆

```python
# 用中文音色说英文
for i, output in enumerate(cosyvoice.inference_cross_lingual(
    '<|en|>Hello, this is a cross-lingual speech synthesis example.',
    'reference_audio.wav')):
    torchaudio.save(f'cross_lingual_{i}.wav', output['tts_speech'], cosyvoice.sample_rate)
```

#### 4.4.4 指令控制（情绪/语速/方言）

```python
# CosyVoice 3.0 指令控制
for i, output in enumerate(cosyvoice.inference_instruct2(
    '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。',
    'You are a helpful assistant. 请用四川话说这句话。<|endofprompt|>',
    'reference_audio.wav', stream=False)):
    torchaudio.save(f'instruct_{i}.wav', output['tts_speech'], cosyvoice.sample_rate)
```

#### 4.4.5 声音转换（Voice Conversion）

```python
# 将一段音频的音色转换为另一段音频的音色
for i, output in enumerate(cosyvoice.inference_vc(
    'source_audio.wav',    # 源音频（提供内容）
    'reference_audio.wav'  # 参考音频（提供音色）
)):
    torchaudio.save(f'vc_{i}.wav', output['tts_speech'], cosyvoice.sample_rate)
```

### 4.5 CPU 运行示例

如果只有 CPU，可以按以下步骤运行：

```bash
# 1. 安装 CPU 版 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. 安装必要依赖
pip install hyperpyyaml diffusers omegaconf conformer onnx onnxruntime openai-whisper pyworld pyarrow gdown soundfile

# 3. 运行推理（使用 soundfile 保存音频）
python3 -c "
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import soundfile as sf

cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT', fp16=False)
for i, j in enumerate(cosyvoice.inference_sft('你好，世界！', '中文女', stream=False)):
    sf.write(f'output_{i}.wav', j['tts_speech'].squeeze().numpy(), cosyvoice.sample_rate)
    print(f'Saved output_{i}.wav')
"
```

**实测性能**（ARM64 CPU, CosyVoice-300M-SFT）：
- 模型加载：3.3 秒
- 生成 4.9 秒音频：29.8 秒（RTF ≈ 6.06）

### 4.6 Web UI 演示

```bash
python webui.py --port 8000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

然后在浏览器中访问 `http://localhost:8000`。

### 4.6 vLLM 加速推理

```bash
# 创建独立环境
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4

# 运行示例
python vllm_example.py
```

### 4.7 服务部署

#### Docker + gRPC

```bash
cd runtime/python
docker build -t cosyvoice:v1.0 .
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 \
  /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && \
  python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"

# 客户端调用
cd grpc && python3 client.py --port 50000 --mode zero_shot
```

#### Docker + FastAPI

```bash
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 \
  /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && \
  python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"

cd fastapi && python3 client.py --port 50000 --mode zero_shot
```

#### TensorRT-LLM（4x 加速）

```bash
cd runtime/triton_trtllm
docker compose up -d
```

---

## 5. 训练

项目提供了完整的训练脚本，位于 `examples/libritts/cosyvoice{,2,3}/` 目录。

```bash
# 使用 DeepSpeed 训练
python cosyvoice/bin/train.py \
  --model cosyvoice3 \
  --config examples/libritts/cosyvoice3/path.yaml \
  --train_engine deepspeed

# 使用 torch DDP 训练
python cosyvoice/bin/train.py \
  --model cosyvoice3 \
  --config examples/libritts/cosyvoice3/path.yaml \
  --train_engine torch_ddp
```

还支持 GRPO（Group Relative Policy Optimization）强化学习训练：

```bash
cd examples/grpo/cosyvoice2
# 具体训练配置见该目录下的文件
```

---

## 6. 项目目录结构

```
CosyVoice/
├── cosyvoice/                    # 核心源码
│   ├── bin/                      # 训练/导出脚本
│   │   ├── train.py              # 训练入口
│   │   ├── export_jit.py         # JIT 导出
│   │   └── export_onnx.py        # ONNX 导出
│   ├── cli/                      # 高层 API
│   │   ├── cosyvoice.py          # AutoModel 主入口
│   │   ├── frontend.py           # 前端处理
│   │   └── model.py              # 模型封装
│   ├── dataset/                  # 数据加载与处理
│   ├── flow/                     # Flow Matching 模块
│   │   └── DiT/                  # Diffusion Transformer（3.0）
│   ├── hifigan/                  # HiFi-GAN 声码器
│   ├── llm/                      # LLM 模块（Qwen2 骨干）
│   ├── tokenizer/                # 文本/语音 Tokenizer
│   ├── transformer/              # Transformer 基础组件
│   ├── utils/                    # 工具函数
│   └── vllm/                     # vLLM 集成
├── runtime/                      # 部署运行时
│   ├── python/                   # Docker/gRPC/FastAPI
│   └── triton_trtllm/            # TensorRT-LLM 部署
├── examples/                     # 训练示例
│   ├── libritts/                 # LibriTTS 训练配置
│   └── grpo/                     # GRPO 训练配置
├── third_party/
│   └── Matcha-TTS/               # 第三方 Matcha-TTS 代码
├── asset/                        # 示例音频
├── example.py                    # Python 使用示例
├── vllm_example.py               # vLLM 使用示例
├── webui.py                      # Gradio Web 界面
└── requirements.txt              # Python 依赖
```

---

## 7. 常见问题

### Q: 没有 GPU 能运行吗？

可以，但需注意版本兼容性：

**✅ CosyVoice 1.0 (300M) - CPU 测试通过**（ARM64 CPU, Python 3.13, CosyVoice-300M-SFT）：
- **模型加载**：3.3 秒
- **推理耗时**：29.8 秒（生成 4.9 秒音频）
- **RTF（实时率）**：约 6.06（CPU 上约 6 倍实时时间，GPU 上可降至 0.1 以下）
- **安装 CPU 版 PyTorch**：`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`
- **保存音频**：需额外安装 `pip install soundfile`

**⚠️ CosyVoice 3.0 (0.5B) - CPU 兼容性限制**：
- 3.0 模型使用了 BFloat16 精度，在较新的 `transformers` 版本下可能存在类型不匹配问题（`mat1 and mat2 must have the same dtype, but got Float and BFloat16`）
- 建议在 GPU 环境或使用 `transformers==4.51.3` 等指定版本运行 3.0

CPU 可用于 1.0 版本的测试和开发，但 2.0/3.0 推荐使用 GPU 环境。

### Q: 显存不足怎么办？

- 使用 `stream=True` 进行流式推理，降低显存峰值
- 使用更小的模型（CosyVoice 1.0 仅需 ~2GB 显存）
- 启用 `torch.compile` 优化

### Q: 如何添加自定义音色？

使用零样本克隆即可，只需准备 3 秒以上的清晰人声音频：

```python
cosyvoice.add_zero_shot_spk('参考文本', 'audio.wav', 'my_spk_name')
```

### Q: 支持哪些音频格式？

支持 WAV、MP3、FLAC、OGG 等常见格式（通过 torchaudio/soundfile 支持）。

---

## 8. 相关资源

- **在线 Demo**：
  - CosyVoice 3.0：https://funaudiollm.github.io/cosyvoice3/
  - CosyVoice 2.0：https://funaudiollm.github.io/cosyvoice2/
  - CosyVoice 1.0：https://fun-audio-llm.github.io

- **模型下载**：
  - ModelScope：https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
  - HuggingFace：https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

- **论文**：
  - CosyVoice 3.0：https://arxiv.org/pdf/2505.17589
  - CosyVoice 2.0：https://arxiv.org/pdf/2412.10117
  - CosyVoice 1.0：https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf

---

## 9. 许可证

本项目采用 Apache 2.0 许可证，详见 `LICENSE` 文件。

---

*文档生成时间：2026-04-26*
