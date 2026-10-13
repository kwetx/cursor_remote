# ncnn_llm 仓库通读笔记

> 基于本地检出 `futz12/ncnn_llm` 的 readme.md、README_CN.md、CMake、src、examples、export 等整理。  
> 许可证：Apache-2.0 · 上游：https://github.com/futz12/ncnn_llm

---

## 1. 项目定位（与 ncnn 的关系）

在 [Tencent/ncnn](https://github.com/Tencent/ncnn) 之上的轻量 **C++20 本地推理运行时**，面向边缘设备、桌面 CPU、可选 Vulkan GPU。

- **不是**单一聊天 demo，而是可复用运行时库 + 一批 CLI
- 覆盖：LLM、VLM、OCR、ASR、判别器（Laya）、文本/多模态 Embedding
- `ncnn` 以 **git submodule** 内嵌（`.gitmodules`）；也可 `USE_SYSTEM_NCNN=ON`
- 渊源：由 nihui 的 ncnn `kvcache` 实验扩展而来
- 构建系统：**CMake**（官方与 CI 均为 CMake；无正式 xmake）

平台声明：Windows / Linux / Android；CI 覆盖 Ubuntu / macOS / Windows。

---

## 2. 核心特性

| 能力 | 说明 | 主要代码 |
|------|------|----------|
| 统一 LLM/VLM | KV-cache 自回归；CPU + 可选 Vulkan；采样/工具调用 | `src/ncnn_llm_gpt.*` |
| Qwen3.5 混合注意力 | GatedDeltaRule + ShortConv；x86 SIMD | `src/kernel/` |
| OCR | GLM-OCR / HunyuanOCR | `src/ncnn_llm_ocr.*` |
| ASR | Qwen3-ASR（PCM→mel→encoder→decoder） | `src/ncnn_llm_asr.*` |
| 判别器 Laya | backbone + scorer + act_head | `src/ncnn_llm_laya.*` |
| Embedding | Jina Nano 文本、Jina-CLIP-v2 图文 | `src/ncnn_embedding.*` |
| 分词 | BPE / BBPE、Unigram | `src/utils/tokenizer/` |
| INT8 块量化 | Decoder 等量化；LM Head/Embedding 常保留高精度 | `export/quantize_model.py` |

**README 支持模型（节选）**

- LLM：YoutuLLM、MiniCPM4/5、Qwen3、Qwen3.5  
- VLM：Qwen2.5-VL、Qwen3.5-VL  
- OCR：GLM-OCR、HunyuanOCR  
- ASR：Qwen3 ASR  
- Embedding：Jina-Embeddings-v5-Text-Nano、Jina-CLIP-v2  

---

## 3. 目录结构

```text
ncnn_llm/
├── CMakeLists.txt / cmake/    # 工程与依赖（ncnn、nlohmann_json）
├── ncnn/                      # 上游 ncnn submodule（需 --recursive）
├── src/                       # 静态库：gpt/ocr/asr/laya/embedding + kernel/utils
├── examples/                  # llm_ncnn_run、ocr/asr/laya/embedding/clip CLI
├── export/                    # Python 导出、KV-cache 改写、BF16/INT8
├── benchmark/                 # benchllm
├── tests/                     # test_llm / test_bf16 / test_kernel 等
├── assets/                    # 模型目录约定位置（权重通常不入库）
└── readme.md / README_CN.md
```

**本地注意**：浅克隆若未 `submodule update --init --recursive`，`ncnn/` 可能为空；完整权重需从模型库下载或自行导出。

---

## 4. 构建与运行

### 依赖

| 项 | 要求 |
|----|------|
| 编译器 | C++20 |
| CMake | ≥ 3.15 |
| OpenMP | 必需 |
| Vulkan SDK | 可选（`NCNN_LLM_ENABLE_VULKAN`，默认常 ON） |
| nlohmann_json | 系统包或 FetchContent |
| 量化工具 | `-DNCNN_LLM_ENABLE_TOOLS=ON` 以编出 `ncnnllm2int` 等 |

### 构建

```bash
git clone --recursive https://github.com/futz12/ncnn_llm.git
# 或：git submodule update --init --recursive

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_LLM_ENABLE_VULKAN=ON \
  -DNCNN_LLM_ENABLE_TOOLS=ON

cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

### 主要产物

- 库：`ncnn_tokenizer`、`ncnn_llm`（静态）
- 程序：`llm_ncnn_run`、`benchllm`、`ocr_main`、`asr_main`、`laya_main`、`embedding_main`、`clip_main` 等

模型镜像参考：https://mirrors.sdu.edu.cn/ncnn_modelzoo/  
目录需含 `model.json`、`*.ncnn.param` / `*.ncnn.bin`、分词文件。

---

## 5. 导出与量化

配置约定：各模型目录下的 `model.json`（`model_type`、params、tokenizer、setting）。

| 脚本 | 用途 |
|------|------|
| `llama_export.py` | LLaMA 系（如 MiniCPM5） |
| `qwen35_export.py` / `qwen35_vision_export.py` | Qwen3.5 文本/视觉塔 |
| `laya_export.py` | Laya 判别器 |
| `add_kvcache.py` | SDPA → 增量 KV-cache |
| `quantize_model.py` | 目录级 INT8（调 `ncnnllm2int`） |
| `bf16_convert.py` | `ncnnoptimize` → BF16 |

典型流水线：HF → pnnx/trace → 加 KV-cache → BF16 → INT8（LM Head / Embedding 常不量化）。

```bash
python export/quantize_model.py --model ./assets/qwen3_0.6b \
  --output ./assets/qwen3_0.6b_int8 --bits 8 --block 64
./build/llm_ncnn_run --model ./assets/qwen3_0.6b_int8 --threads 8
```

说明：Gemm block INT8 主要优化 **CPU**；量化层会走 CPU，Vulkan 收益有限。旧导出脚本可能过时，以最新 `model.json` 与示例为准。

---

## 6. 示例用法

```bash
# 主 CLI：文本 / VLM
./build/llm_ncnn_run --model ./assets/qwen3_0.6b
./build/llm_ncnn_run --model ./assets/qwen3_0.6b --use-vulkan --threads 4
./build/llm_ncnn_run --model ./assets/qwen2.5_vl_3b --image ./assets/test.jpg

# OCR / ASR / Embedding / Laya
./build/ocr_main --model ./assets/glm_ocr --image ./test_ocr.png
./build/asr_main --audio ./audio.wav --model ./assets/qwen3_asr_0.6b   # 期望 16kHz
./build/embedding_main ./assets/jina-embeddings-v5-text-nano
./build/laya_main --model ./assets/laya_multilingual_int8 \
  --json ./examples/laya_multilingual_request.json
```

C++ API 要点：`ncnn_llm_gpt`（`prefill` / `generate`）、OCR/ASR 各自 prefill+generate、Embedding 的 `encode_text` / `encode_image_file`。

---

## 7. 适用场景与局限

**适合**

- 端侧/桌面无 Python 服务的 C++ 本地推理
- 小～中等参数量聊天、VLM、OCR、ASR、意图判别、检索嵌入
- 需要 INT8 降内存 + 提 CPU 吞吐，并复用 ncnn KV-cache / SDPA

**局限**

1. 必须初始化 ncnn 子模块，否则 CMake 失败  
2. 权重不在仓库内（`assets/` 多被 ignore）  
3. 构建只有 CMake；模型族需专用导出脚本，非任意 HF LLM  
4. INT8 加速偏 CPU；ASR 示例假定 16 kHz  
5. 部分示例（如部分 tokenizer main）可能未接入 CMake  
6. 路线图仍在扩架构、提升 Vulkan/CPU、完善导出文档  

---

## 8. 一句话总结

`ncnn_llm` 是挂在官方 ncnn 上的多模态本地推理运行时：CMake 产出静态库与 CLI，用 `model.json` + 转换模型目录驱动 LLM/VLM/OCR/ASR/Laya/Embedding；适合端侧部署，完整使用依赖子模块、外部权重，以及（导出时）ncnn 量化工具链。

---

## 9. 关键入口

| 用途 | 路径 |
|------|------|
| 英文/中文 README | `readme.md`、`README_CN.md` |
| 构建 | `CMakeLists.txt`、`cmake/deps_ncnn.cmake` |
| 核心运行时 | `src/ncnn_llm_gpt.*`、`src/ncnn_text_runtime.*` |
| 主 CLI | `examples/llm_ncnn_run/` |
| 导出 | `export/` |
| 模型镜像 | https://mirrors.sdu.edu.cn/ncnn_modelzoo/ |
