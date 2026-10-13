# MNN 仓库通读笔记

> 基于本地检出 `alibaba/MNN` 的 README、README_CN、AGENTS.md、docs、transformers、source 等整理。  
> 许可证：Apache-2.0 · 上游：https://github.com/alibaba/MNN  
> 版本线索：`include/MNN/MNNDefine.h`（如 3.6.x）

---

## 1. 项目定位

**MNN（Mobile Neural Network）** 是阿里开源的轻量级深度神经网络引擎，面向服务器 / PC / 手机 / 嵌入式。

- 主战场：**端侧高性能推理**（速度 + 体积）
- 同时提供：训练相关能力、模型压缩、通用张量计算（Express）
- 阿里内部：淘宝、天猫、优酷等 30+ App，直播、搜推、以图搜图等大量场景
- 文档站：https://mnn-docs.readthedocs.io ；工作台：http://www.mnn.zone

`AGENTS.md` 对贡献者强调「轻量推理、优先性能与二进制体积」；对外 README 仍写支持推理与训练——大模型训练主战场仍建议 PyTorch/TF，MNN 偏端侧微调与部署。

---

## 2. 核心特性

| 维度 | 要点 |
|------|------|
| 轻量 | 主体无重依赖；Android 核心 so 约百 KB～MB 级；`MNN_BUILD_MINI` 可再裁剪 |
| 通用 | TF / Caffe / ONNX / TorchScript / TFLite；CNN、RNN、GAN、Transformer |
| 高性能 | ARM/x86 SIMD 与汇编；Winograd 等；Metal / OpenCL / Vulkan / CUDA |
| 后端扩展 | Hexagon、QNN、CoreML、HIAI、NNAPI、RKNN、TensorRT 等 |
| 易用 | Express（类 numpy）、MNN-CV、pymnn、端侧训练/QLoRA |
| LLM | 完整导出 → 引擎 → Chat App / mnncli 闭环 |

**API 两层**

- 低层 Session：`Interpreter → createSession → runSession`
- 高层 Module（推荐，LLM/Diffusion）：`Module::load → onForward(VARP)`

内部张量常用 **NC4HW4**；算子由 FlatBuffers schema（`schema/default/`）定义。

---

## 3. 目录结构

```text
MNN/
├── include/MNN/     # 对外 C++ 头文件
├── source/          # 推理核心 + 各 backend
├── express/         # 动态图 / Module / 控制流
├── schema/          # FlatBuffers（勿碰 private/）
├── tools/           # converter、compress、quantization、train、cv、audio
├── transformers/    # llm / diffusion 导出与引擎
├── pymnn/           # Python 绑定
├── apps/            # Android/iOS Chat、TaoAvatar、Sana、mnncli
├── docs/            # Sphinx 文档源
├── demo/ benchmark/ test/ project/
├── skills/          # Agent 技能（新 LLM、新 Op、优化等）
└── AGENTS.md / CLAUDE.md
```

| 路径 | 作用 |
|------|------|
| `source/core/` | Interpreter、Session、Pipeline、KVCacheManager |
| `source/backend/` | cpu、metal、opencl、vulkan、cuda、hexagon、qnn… |
| `transformers/llm/` | export / engine / finetune / eval |
| `apps/Android/MnnLlmChat/` 等 | 上层多模态应用 |

---

## 4. 构建与使用要点

```bash
mkdir build && cd build && cmake .. && make -j$(nproc)

# LLM（会连带 LOW_MEMORY + TRANSFORMER_FUSE）
cmake .. -DMNN_BUILD_LLM=ON
```

**重要选项**

- 功能：`MNN_BUILD_CONVERTER`、`MNN_BUILD_LLM`、`MNN_BUILD_LLM_OMNI`、`MNN_BUILD_DIFFUSION`、`MNN_BUILD_TRAIN`、`MNN_LOW_MEMORY`、`MNN_SUPPORT_TRANSFORMER_FUSE`
- 后端：`MNN_METAL`、`MNN_OPENCL`、`MNN_VULKAN`、`MNN_CUDA`、`MNN_ARM82`、`MNN_HEXAGON`、`MNN_QNN` 等
- 体积：`MNN_BUILD_MINI` / `MNN_REDUCE_SIZE`

默认产物常含 `libMNN.so`、`express/libMNN_Express.so`。静态库需 whole-archive / force_load，否则自注册算子可能丢失。

推理流程：外框训练或 MNN-Train → `MNNConvert` → Session/Module；含控制流须用 Module。

---

## 5. 模型转换与 transformers

### 通用转换

- 工具：`tools/converter/` → `MNNConvert`（`-DMNN_BUILD_CONVERTER=ON`）
- 能力：图优化、`--fp16`、权重量化、`--transformerFuse`、正确性测试等
- 压缩：`tools/mnncompress/`

### MNN-LLM 导出

```bash
cd transformers/llm/export
python llmexport.py --path /path/to/HF-model --export mnn --hqq
```

典型产物：`config.json`、`llm.mnn`、`llm.mnn.weight`、`llm_config.json`、tokenizer、可选 embeddings。

量化：`--quant_bit` / `--quant_block`、`--hqq` / `--awq` / GPTQ / LoRA 合并等。另有 `safetensors2mnn.py`、`gguf2mnn.py`。

### Diffusion

`transformers/diffusion/`：HF → ONNX → MNN；需 `-DMNN_BUILD_DIFFUSION=ON`。

预转换模型：ModelScope `MNN` 组织、Hugging Face `taobao-mnn`。

---

## 6. Python 与上层应用

### pymnn

- `pip install MNN`，或 `pymnn/pip_package` 本地构建
- 绑定覆盖 expr、nn、cv、audio、llm、compress、optim 等

### Express

动态图与通用计算：`VARP` / `Expr` / `Executor` / `Module`；控制流含 If / While / MoE / Pipeline。

### 上层应用（`apps/`）

| 应用 | 说明 |
|------|------|
| MNN Chat（Android/iOS） | 多模态对话、文生图 |
| TaoAvatar | 端侧 LLM+ASR+TTS+3D 数字人（旗舰机） |
| Sana | 卡通风格图像编辑 |
| mnncli | CLI：下载/运行/benchmark、OpenAI 兼容 serve |

---

## 7. LLM / 大模型能力

- **导出**：`transformers/llm/export/`（`llmexport.py`、`model_mapper.py`）
- **引擎**：`engine/src/llm.cpp`（文本）、`omni.cpp`（视觉/音频）；tokenizer、sampler、KVCache
- **微调**：`finetune/mnn_qlora.py`（QLoRA/HQQ）
- **编译**：`-DMNN_BUILD_LLM=ON`；Omni 再加 `-DMNN_BUILD_LLM_OMNI=ON`

已映射模型族（节选）：llama、qwen/qwen2/qwen3 系列（含 VL/Omni/ASR/TTS/MoE）、baichuan、chatglm、phi、internlm、internvl、gemma、deepseek-vl、minicpm、smolvlm、mobilellm 等。

运行时能力：cpu/opencl/metal、mmap/kvcache_mmap、Flash Attention、多种采样、投机解码、多 LoRA、`<img>`/`<audio>` 多模态 prompt。

快速跑：`./llm_demo <model_dir>/config.json`。

---

## 8. 适用场景与局限

**适合**

- 移动/嵌入式 CNN 与轻量 Transformer
- 端侧 LLM / VL / Omni / Diffusion（量化 + 低内存 + 可选 GPU）
- 包体积敏感、隐私优先本地推理
- 跨平台统一运行时（含 Harmony / WASM）

**局限**

- 非云端大规模训练框架
- LLM 对设备要求高（Chat/TaoAvatar 偏旗舰机）
- NPU 成熟度不均，常需额外 SDK
- `MNN_BUILD_MINI` 等裁剪后仅适合受限模型
- 权重量化默认偏体积，要速度还需配合运行时低内存/动态量化路径
- `schema/private/`、`source/internal/` 不对外部贡献开放

---

## 9. 关键入口

| 用途 | 路径 |
|------|------|
| 中文总览 | `README_CN.md` |
| 贡献/架构指引 | `AGENTS.md` |
| CMake 选项 | `CMakeLists.txt`、`docs/compile/cmake.md` |
| LLM 文档 | `docs/transformers/llm.md`、`transformers/README.md` |
| 导出 | `transformers/llm/export/llmexport.py` |
| Python | `pymnn/` |
| Chat App | `apps/Android/MnnLlmChat/` |
