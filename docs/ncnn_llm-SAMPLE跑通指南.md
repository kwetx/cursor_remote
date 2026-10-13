# ncnn_llm Sample 跑通指南

本文记录一次完整跑通 `llm_ncnn_run` + `qwen3_0.6b_int8` 的步骤：环境、拉代码、拉模型、编译、运行，以及相关脚本/代码各自在干什么。

实测成功回复示例：

> User: 用一句话介绍你自己  
> Assistant: 我是热爱学习和探索的AI助手，在这里与你一起发现世界的新奇之处！

---

## 0. 目标与产物

| 目标 | 说明 |
|------|------|
| 可执行文件 | `build/llm_ncnn_run` |
| 模型目录 | `assets/qwen3_0.6b_int8/` |
| 运行方式 | 标准输入喂问题，或交互式聊天 |

本指南走的是「**直接下载已转换好的 ncnn 模型 → 编译运行时 → CLI 推理**」路径。  
若要从 HuggingFace 自行导出，见文末「可选：自己导出/量化」。

---

## 1. 环境依赖

### 1.1 需要安装什么

| 依赖 | 用途 |
|------|------|
| C++20 编译器（GCC 10+ / Clang 11+） | 编译 `ncnn_llm` |
| CMake ≥ 3.15 | 生成构建系统 |
| OpenMP（`libomp-dev` / `libgomp`） | 多线程推理 |
| `nlohmann-json3-dev`（或构建时 FetchContent） | 读 `model.json`、工具调用 JSON |
| `build-essential` / `libstdc++-*-dev` | 链接 `libstdc++` |
| curl / wget | 下载模型文件 |
| git | 克隆仓库与子模块 |
| Vulkan SDK（可选） | GPU；本 sample 关闭 Vulkan |

Ubuntu 示例：

```bash
sudo apt-get update
sudo apt-get install -y build-essential g++ cmake git curl \
  libomp-dev nlohmann-json3-dev
# 若链接报 cannot find -lstdc++，再装：
sudo apt-get install -y libstdc++-14-dev
```

### 1.2 资源建议

- `qwen3_0.6b_int8` 权重约 **744MB**（decoder ~447MB + embed ~297MB）
- 推理时建议可用内存 ≥ **4–5GB**（模型 + KV cache + 系统）
- 磁盘：源码 + 构建 + 模型合计约数 GB

---

## 2. 拉取代码与 ncnn 子模块

### 2.1 克隆

```bash
git clone --recursive https://github.com/futz12/ncnn_llm.git
cd ncnn_llm
```

若已经克隆但未带子模块：

```bash
git submodule update --init --recursive
```

### 2.2 这一步在干什么

| 对象 | 作用 |
|------|------|
| `ncnn_llm` 本体 | LLM/VLM/OCR 等运行时 + CLI 示例 |
| 子模块 `ncnn/` | 上游 [Tencent/ncnn](https://github.com/Tencent/ncnn)，**必须用仓库锁定的 commit**（例如 `ab44b901…`） |
| `ncnn/glslang` | Vulkan 着色器编译（关 Vulkan 时仍可能被拉取，但不使用） |

**注意**：不要随便把任意版本的 ncnn 软链进来。版本不对会出现缺少 `LayerTypeStat` / `reset_layer_benchmark` 等 API，导致编译失败。

相关文件：

- `.gitmodules`：声明子模块 URL
- `cmake/deps_ncnn.cmake`：检测并编译捆绑的 ncnn，或改用系统 ncnn（`USE_SYSTEM_NCNN`）

---

## 3. 拉取模型文件

### 3.1 模型从哪来

官方镜像（已转换好的 ncnn 目录）：

https://mirrors.sdu.edu.cn/ncnn_modelzoo/

本 sample 使用最小可用 LLM 之一：

https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3_0.6b_int8/

### 3.2 下载命令

```bash
mkdir -p assets/qwen3_0.6b_int8
BASE=https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3_0.6b_int8
cd assets/qwen3_0.6b_int8

# 元数据 / 分词 / 图结构（小文件）
for f in model.json \
         qwen3_decoder.ncnn.param \
         qwen3_embed_token.ncnn.param \
         qwen3_proj_out.ncnn.param \
         merges.txt vocab.txt; do
  curl -L --fail -o "$f" "$BASE/$f"
done

# 权重（大文件）
for f in qwen3_decoder.ncnn.bin qwen3_embed_token.ncnn.bin; do
  curl -L --fail --retry 3 -o "$f" "$BASE/$f"
done

cd ../..
ls -lh assets/qwen3_0.6b_int8/
```

### 3.3 目录里每个文件干什么

| 文件 | 作用 |
|------|------|
| `model.json` | 总配置：模型类型、各网络文件名、tokenizer、RoPE、量化信息 |
| `*.ncnn.param` | ncnn 网络结构（层名、连接、超参） |
| `*.ncnn.bin` | 权重二进制 |
| `vocab.txt` / `merges.txt` | BBPE 分词词表与 merge 规则 |

`model.json` 里关键字段含义：

| 字段 | 含义 |
|------|------|
| `type: "qwen3"` | 选用 Qwen3 chat 模板 |
| `params.decoder_*` | Transformer decoder 图 |
| `params.embed_token_*` | token embedding |
| `params.proj_out_*` | LM Head；本模型 **权重共享**，`proj_out_bin` 指向 `qwen3_embed_token.ncnn.bin` |
| `tokenizer.type: "bbpe"` | 字节级 BPE |
| `quantization` | decoder 做了 INT8 block 量化 |

因此镜像里**没有**单独的 `qwen3_proj_out.ncnn.bin` 是正常的（404），不是缺文件。

下载后目录应类似：

```text
assets/qwen3_0.6b_int8/
├── model.json
├── merges.txt
├── vocab.txt
├── qwen3_decoder.ncnn.param / .bin
├── qwen3_embed_token.ncnn.param / .bin
└── qwen3_proj_out.ncnn.param          # bin 与 embed 共享
```

---

## 4. 编译

### 4.1 配置 + 构建

```bash
# 本 sample：关 Vulkan，只编运行示例，加快构建
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_LLM_ENABLE_VULKAN=OFF \
  -DNCNN_LLM_ENABLE_TOOLS=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_BUILD_TESTS=OFF

cmake --build build --config Release --target llm_ncnn_run -j$(nproc)
```

成功后应有：

```bash
ls -lh build/llm_ncnn_run
```

### 4.2 CMake / 构建相关文件在干什么

| 文件/目标 | 作用 |
|-----------|------|
| `CMakeLists.txt` | 工程入口：C++20、OpenMP、编静态库与可执行文件 |
| `cmake/deps_ncnn.cmake` | 编译子模块 ncnn，或找系统 libncnn |
| `cmake/deps_json.cmake` | 找系统 nlohmann_json，没有则 FetchContent |
| 目标 `ncnn` | 上游推理引擎静态库 |
| 目标 `ncnn_tokenizer` | BPE / Unigram 分词实现 |
| 目标 `ncnn_llm` | 核心运行时（GPT/OCR/ASR/Embedding…） |
| 目标 `llm_ncnn_run` | 聊天 CLI sample |

常用开关：

| 选项 | 含义 |
|------|------|
| `NCNN_LLM_ENABLE_VULKAN` | 是否开 GPU |
| `NCNN_LLM_ENABLE_TOOLS` | 是否编 `ncnnllm2int` / `ncnnoptimize`（导出量化用） |
| `USE_SYSTEM_NCNN` | 不用子模块，改用系统 ncnn |

---

## 5. 运行 sample

### 5.1 交互式

```bash
./build/llm_ncnn_run --model ./assets/qwen3_0.6b_int8 --threads 2 --no-builtin-tools
```

输入问题后回车；输入 `exit` 或 `quit` 结束。

### 5.2 非交互（脚本喂入）

```bash
printf '用一句话介绍你自己\nexit\n' | ./build/llm_ncnn_run \
  --model ./assets/qwen3_0.6b_int8 \
  --threads 2 \
  --max-new-tokens 128 \
  --no-builtin-tools
```

带性能报告：

```bash
printf '用一句话介绍你自己\nexit\n' | ./build/llm_ncnn_run \
  --model ./assets/qwen3_0.6b_int8 \
  --threads 2 \
  --max-new-tokens 64 \
  --no-builtin-tools \
  --perf
```

### 5.3 常用 CLI 参数

| 参数 | 作用 |
|------|------|
| `--model <path>` | 模型目录（含 `model.json`） |
| `--threads <n>` | CPU 线程数 |
| `--max-new-tokens <n>` | 最多生成多少 token |
| `--no-builtin-tools` | 关掉内置计算器/随机数工具演示 |
| `--use-vulkan` / `--vulkan` | 开 GPU（需编译时打开 Vulkan） |
| `--enable-thinking` | 输出 `<think>…</think>` |
| `--perf` / `--perf-level` | 打印 prefill/decode 耗时 |

---

## 6. Sample 代码结构：谁在干什么

运行入口在 `examples/llm_ncnn_run/`，核心库在 `src/`。

### 6.1 CLI 示例（你直接跑的程序）

| 文件 | 大概干什么 |
|------|------------|
| `main.cpp` | 解析参数 → 规范化模型路径 → 构造 `ncnn_llm_gpt` → 可选读图 → 进入 `run_cli` |
| `options.cpp` / `options.h` | 定义并解析上述 CLI 选项 |
| `cli_runner.cpp` | 交互循环：读用户输入 → chat 模板 → `prefill` → `generate` 流式打印 |
| `tools.cpp` | 内置 tool 定义与路由（随机数、加法等演示） |
| `json_utils.cpp` | JSON / 模板相关小工具 |
| `util.cpp` | 读图到 `ncnn::Mat` 等杂项 |

`cli_runner.cpp` 主流程（简化）：

1. 读 `model.json` 的 `type`，选 Chat 模板（Qwen3 / MiniCPM5 / …）
2. 用 system prompt 做一次初始 `prefill`
3. 循环：`User:` → 套模板 → `prefill` → `Assistant:` → `generate`（token 回调打印）
4. 若有 tool call，走 callback 再继续生成

### 6.2 核心运行时库

| 文件 | 大概干什么 |
|------|------------|
| `src/ncnn_llm_gpt.*` | LLM/VLM 主类：加载三份网络、KV cache 解码、工具调用 |
| `src/ncnn_text_runtime.*` | 共享文本路径：embed → decoder → lm_head → 采样 |
| `src/sampling.*` | 采样：argmax / temperature / top-k / top-p / repetition penalty |
| `src/utils/tokenizer/*` | BPE、Unigram 实现 |
| `src/utils/prompt.*` | ChatML / Qwen3 / MiniCPM5 等模板拼接 |
| `src/utils/rope_embed.*` / `vision_rope.*` | RoPE / 视觉 RoPE |
| `src/utils/perf_profiler.*` | `--perf` 性能统计 |
| `src/kernel/gdr.*` / `shortconv.*` / `lm_head.*` | Qwen3.5 等自定义算子与 LM Head 加速 |

加载时大致会：

1. 解析 `model.json`
2. 加载 decoder / embed / proj_out 的 param+bin
3. 若 `proj_out_bin` 与 embed 相同，则 **权重共享**，省一份大文件内存
4. 初始化 tokenizer 与采样配置

### 6.3 推理时发生了什么（概念）

```text
用户文本
  → tokenizer.encode
  → chat 模板（加 system / special tokens）
  → embed 查表
  → decoder prefill（整段 prompt，写 KV cache）
  → lm_head + 采样得到第一个新 token
  → 循环 decode：单 token → decoder step → lm_head → 采样
  → tokenizer.decode 流式输出
  → 遇到 eos / 达到 max_new_tokens 结束
```

---

## 7. 可选：自己导出 / 量化（本 sample 未走）

若不用镜像里的成品模型，而是从 HuggingFace 自己转：

| 脚本 | 作用 |
|------|------|
| `export/llama_export.py` | LLaMA 系（如 MiniCPM）HF → pnnx/ncnn → KV-cache → BF16/INT8 |
| `export/qwen35_export.py` | Qwen3.5 文本塔导出 |
| `export/qwen35_vision_export.py` | Qwen3.5 视觉塔 |
| `export/add_kvcache.py` | 把 SDPA 图改成增量 KV-cache |
| `export/bf16_convert.py` | 调 `ncnnoptimize` 转 BF16 |
| `export/quantize_model.py` | 调 `ncnnllm2int` 做 INT8 block 量化 |
| `export/extract_tokenizer.py` | 从 HF tokenizer 抽 vocab/merges |

量化成品目录示例：

```bash
# 需编译时打开工具：-DNCNN_LLM_ENABLE_TOOLS=ON
python export/quantize_model.py \
  --model ./assets/qwen3_0.6b \
  --output ./assets/qwen3_0.6b_int8 \
  --bits 8 --block 64

./build/llm_ncnn_run --model ./assets/qwen3_0.6b_int8 --threads 8
```

---

## 8. 一键复现清单（最短路径）

```bash
# 1) 代码
git clone --recursive https://github.com/futz12/ncnn_llm.git
cd ncnn_llm

# 2) 依赖（Ubuntu）
sudo apt-get install -y build-essential g++ cmake git curl \
  libomp-dev nlohmann-json3-dev libstdc++-14-dev

# 3) 模型
mkdir -p assets/qwen3_0.6b_int8
BASE=https://mirrors.sdu.edu.cn/ncnn_modelzoo/qwen3_0.6b_int8
cd assets/qwen3_0.6b_int8
for f in model.json qwen3_decoder.ncnn.param qwen3_embed_token.ncnn.param \
         qwen3_proj_out.ncnn.param merges.txt vocab.txt \
         qwen3_decoder.ncnn.bin qwen3_embed_token.ncnn.bin; do
  curl -L --fail --retry 3 -o "$f" "$BASE/$f"
done
cd ../..

# 4) 编译
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_LLM_ENABLE_VULKAN=OFF
cmake --build build --config Release --target llm_ncnn_run -j$(nproc)

# 5) 运行
printf '用一句话介绍你自己\nexit\n' | ./build/llm_ncnn_run \
  --model ./assets/qwen3_0.6b_int8 \
  --threads 2 \
  --max-new-tokens 128 \
  --no-builtin-tools
```

---

## 9. 常见问题

| 现象 | 原因 / 处理 |
|------|-------------|
| `cannot find -lstdc++` | 安装 `libstdc++-*-dev` / `build-essential` |
| `LayerTypeStat` / `reset_layer_benchmark` 编译错误 | ncnn 子模块版本不对 → `git submodule update --init --recursive` |
| `proj_out.ncnn.bin` 404 | 正常，本模型与 embed 权重共享 |
| Assistant 几乎无输出 | 可能耗在空 think / 工具调用；加 `--no-builtin-tools`，适当增大 `--max-new-tokens` |
| 内存不够 | 用 `*_int8` 模型、减小 threads、关其它进程 |
| OpenMP 找不到 | 安装 `libomp-dev` |

---

## 10. 相关链接

- 仓库：https://github.com/futz12/ncnn_llm  
- 模型库：https://mirrors.sdu.edu.cn/ncnn_modelzoo/  
- 上游 ncnn：https://github.com/Tencent/ncnn  
- 本环境实测日志：仓库内 `docs/ncnn_llm_sample_run.log`（若在本 monorepo 中）
