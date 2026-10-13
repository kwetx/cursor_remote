# ncnn 仓库通读笔记

> 基于本地检出 `Tencent/ncnn` 的 README、docs、CMake、src/tools/python 等整理。  
> 许可证：BSD 3-Clause · 上游：https://github.com/Tencent/ncnn

---

## 1. 项目定位

面向**移动端 / 嵌入式 / 桌面**的高性能神经网络**推理**框架（不做训练）。

起源问题：模型要在手机落地，Caffe 依赖重、手机无 CUDA。目标是「又快又小」的前向实现：

- 无第三方运行时依赖
- CPU + Vulkan GPU
- 通过 **pnnx** 等工具把 PyTorch / ONNX 转成 ncnn 模型

腾讯内部已用于 QQ、Qzone、微信、天天 P 图等。推荐路径：**PyTorch → pnnx → ncnn**。

---

## 2. 核心特性

| 维度 | 要点 |
|------|------|
| 依赖 | 运行时零第三方依赖（不依赖 BLAS / NNPACK） |
| API | 纯 C++、C API、Python（pybind11） |
| CPU | ARM NEON/SVE、x86 SSE/AVX/AVX512、MIPS MSA、RISC-V RVV、LoongArch 等 |
| GPU | Vulkan（`NCNN_VULKAN`，子模块 `glslang`） |
| 内存 | 显式 blob/workspace allocator，可裁剪极简/裸机构建 |
| 精度 | FP16、INT8 量化、weight quant、BF16 |
| 模型 | `.param` + `.bin`，支持文件/内存加载 |
| Transformer | KV Cache（MultiHeadAttention / SDPA），见 `docs/developer-guide/kvcache.md` |

覆盖分类、检测、分割、姿态、OCR 等；示例还扩展到 Whisper、Piper 等；benchmark 含 LLM 相关项。

---

## 3. 目录结构

```text
ncnn/
├── CMakeLists.txt / cmake/ / toolchains/   # 构建与交叉编译
├── src/          # 推理核心：Net/Mat/Layer/GPU/Allocator + 各 ISA 实现
├── tools/        # 转换、优化、量化（含 pnnx、onnx、quantize 等）
├── examples/     # C++ 示例（YOLO、SCRFD、PP-OCR、Whisper 等）
├── python/       # Python 绑定、model_zoo
├── benchmark/    # benchncnn / benchncnn_llm
├── docs/         # 构建、用法、FAQ、开发者指南
└── tests/        # 算子/单元测试
```

- `src/`：约百余层通过 `ncnn_add_layer` 注册；Vulkan shader 在 `layer/vulkan/`
- `docs/`：`Home.md`、`how-to-build/`、`how-to-use-and-FAQ/`、`developer-guide/`、`benchmark/`

---

## 4. 构建与使用要点

### 平台

Android / HarmonyOS / iOS / macOS / Linux（多 arch）/ Windows / WebAssembly / Raspberry Pi / Jetson / QNX / ESP32 / 海思 等；FAQ 称支持裸机。

### 典型构建

```bash
git clone --recursive https://github.com/Tencent/ncnn.git
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..
cmake --build . -j$(nproc)
```

### 关键 CMake 选项

- 功能：`NCNN_VULKAN`、`NCNN_OPENMP`、`NCNN_PYTHON`、`NCNN_INT8`、`NCNN_C_API`
- 产物：`NCNN_BUILD_TOOLS` / `EXAMPLES` / `BENCHMARK` / `TESTS`
- 极简：`NCNN_SIMPLEOCV` / `SIMPLEOMP` / `SIMPLESTL`，以及按层 `-DWITH_LAYER_<name>=OFF`

### 推理骨架

`ncnn::Net` → `load_param` / `load_model` → `create_extractor` → `input` / `extract`。

---

## 5. 模型转换与工具链

| 来源 | 推荐路径 |
|------|----------|
| PyTorch | `pnnx`（`tools/pnnx/`，也可 `pip install pnnx`） |
| ONNX | `pnnx model.onnx`（推荐）；遗留 `onnx2ncnn` |
| 优化 | `ncnnoptimize` |
| 嵌入二进制 | `ncnn2mem` |
| INT8 PTQ | `ncnn2table` → `ncnn2int8`（另有 LLM 相关 `ncnnllm2*`） |
| 遗留 | caffe2ncnn、mxnet2ncnn、darknet2ncnn |

文档明确：传统 `pytorch→onnx→onnx2ncnn` 易因算子/动态 shape 失败，应优先 pnnx。

---

## 6. Python 支持

路径：`python/`。

- 安装：`pip install ncnn`，或源码 `NCNN_PYTHON=ON`
- API 与 C++ 对齐，与 NumPy 可零拷贝互转
- `model_zoo` 含 YOLO、SSD、RetinaFace、NanoDet 等

---

## 7. 适用场景与局限

**适合**

- 包体积/依赖敏感的边侧推理（检测、分类、人脸、OCR、部分语音）
- Vulkan 跨厂商 GPU；WebAssembly；可裁剪裸机
- 算子支持时的 Transformer / 小规模 LLM（需配合社区如 `ncnn_llm`）

**局限**

- 仅推理，不训练；GPU 路径是 Vulkan 而非 CUDA/TensorRT
- ARMv7 NEON 浮点不完全遵循 IEEE754（部分数学为快速实现）
- 新/复杂算子可能需自定义层；极简构建会丢掉文件加载、像素工具或量化能力

---

## 8. 与竞品（文档有提及的部分）

仓库**没有**系统对比 MNN / TFLite / OpenVINO 的专章。文档明确写到的主要是：

- vs caffe-android-lib / mini-caffe：体积、耗时、内存与功耗基准
- vs ONNX 中间表示 / onnx2ncnn：pnnx 更贴 PyTorch、图更干净、可直接出 ncnn

---

## 9. 关键入口

| 用途 | 路径 |
|------|------|
| 总览 | `README.md` |
| 起源与 FAQ | `docs/Home.md` |
| 构建 | `docs/how-to-build/how-to-build.md` |
| CMake 选项 | `CMakeLists.txt` |
| pnnx 转换 | `tools/pnnx/`、`docs/how-to-use-and-FAQ/use-ncnn-with-pytorch-or-onnx.md` |
| 量化 | `tools/quantize/` |
| Python | `python/README.md` |
| KV Cache | `docs/developer-guide/kvcache.md` |
