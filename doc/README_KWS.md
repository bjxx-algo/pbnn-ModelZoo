# KWS — 关键词识别 (Keyword Spotting)

基于 Zipformer2 Transducer 的流式关键词识别，encoder 使用 ONNX Runtime 推理，decoder + joiner 使用 PBNN 加速。

## 模型

| 组件 | 格式 | 说明 |
|------|------|------|
| encoder | `.onnx` | Zipformer2 encoder，chunk-16-left-64 |
| decoder + joiner | `.pbnn` | 打包为单个 PBNN 文件 |
| tokens | `.txt` | 词表文件 |
| keywords | `.txt` | 关键词列表，每行一个关键词 |

当前验证模型：`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`（中英文 3M 参数）

## 编译

### 环境要求

- CMake ≥ 3.16
- aarch64 交叉编译工具链（`aarch64-linux-gnu-g++`）
- ONNX Runtime aarch64 预编译库（位于 `/host/cc_for_aarch64/third_parties/onnxruntime-linux-aarch64`）

### 构建步骤

```bash
cd pbnn-ModelZoo
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake
make -j$(nproc)
```

产物：`build/model-zoo`（aarch64 ELF 可执行文件）

## 运行

### 1. 设置库路径

```bash
export LD_LIBRARY_PATH=/data/fuwj/pbnn-ModelZoo/lib/kws/:/data/fuwj/pbnn-ModelZoo/lib/pb_sdk/:$LD_LIBRARY_PATH
```

### 2. 执行推理

```bash
./build/model-zoo kws \
    --encoder=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/encoder-epoch-13-avg-2-chunk-16-left-64.onnx \
    --decoder=/data/fuwj/kws_deploy_simple/v3.1.1/models/kws.pbnn \
    --joiner=/data/fuwj/kws_deploy_simple/v3.1.1/models/kws.pbnn \
    --tokens=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/tokens.txt \
    --keywords-file=sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/keywords.txt \
    --num-threads=4 \
    input.wav
```

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--encoder` | 是 | — | Encoder ONNX 模型路径 |
| `--decoder` | 是 | — | Decoder PBNN 模型路径 |
| `--joiner` | 是 | — | Joiner PBNN 模型路径（与 decoder 同文件） |
| `--tokens` | 是 | — | 词表文件路径 |
| `--keywords-file` | 是 | — | 关键词文件路径 |
| `--num-threads` | 否 | 1 | 推理线程数 |
| `--sample-rate` | 否 | 16000 | 音频采样率 |
| `--feat-dim` | 否 | 80 | FBank 特征维度 |
| `--max-active-paths` | 否 | 4 | 最大活跃路径数 |
| `--num-trailing-blanks` | 否 | 1 | 尾部空白帧数 |
| `--keywords-score` | 否 | 1.0 | 关键词置信度权重 |
| `--keywords-threshold` | 否 | 0.25 | 关键词检测阈值 |

## 测试结果

平台：Ubuntu 20.04 aarch64，4 线程

| 音频 | 时长 (s) | 耗时 (s) | RTF | 检测结果 |
|------|----------|----------|-----|----------|
| zh_0 | 5.611 | 0.467 | 0.083 | — |
| zh_1 | 5.153 | 0.441 | 0.086 | — |
| zh_2 | 4.524 | 0.386 | 0.085 | — |
| zh_3 | 8.030 | 0.657 | 0.082 | 文森特卡索、法国 |
| zh_4 | 4.599 | 0.394 | 0.086 | 蒋友伯、女儿 |
| zh_5 | 4.153 | 0.374 | 0.090 | 周望军、落实 |
| zh_6 | 3.546 | 0.317 | 0.089 | 朱丽楠、见面会 |
| en_0 | 6.625 | 0.558 | 0.084 | LIGHT_UP |
| en_1 | 16.715 | 1.312 | 0.078 | LOVELY_CHILD |

**平均 RTF ≈ 0.084**，满足实时性要求。

## 输出格式

命中关键词时输出 JSON：

```json
{
  "start_time": 0.00,
  "keyword": "文森特卡索",
  "timestamps": [0.64, 0.76, 0.96, 1.04, 1.28, 1.36, 1.52, 1.64, 1.84, 1.96],
  "tokens": ["w", "én", "s", "ēn", "t", "è", "k", "ǎ", "s", "uǒ"]
}
```

| 字段 | 说明 |
|------|------|
| `start_time` | 音频流起始时间偏移 (秒) |
| `keyword` | 命中的关键词 |
| `timestamps` | 各 token 的时间戳 (秒) |
| `tokens` | 拆分后的 token 序列 |

末尾输出整体统计：`Duration: 8.030s | Elapsed: 0.657s | RTF: 0.082`

## 目录结构

```
pbnn-ModelZoo/
├── CMakeLists.txt
├── toolchain-aarch64.cmake
├── include/
│   ├── kws/kws.h
│   └── pb_sdk/
│       ├── pb_infer_api.h
│       └── qm_runtime.h
├── lib/
│   ├── kws/
│   │   ├── libsherpa-onnx-core.so
│   │   ├── libkaldi-native-fbank-core.so
│   │   └── libkissfft-float.a
│   └── pb_sdk/
│       ├── libpb_inference_engine.so.3.1.1
│       ├── libqm_runtime.so
│       ├── libonnxruntime.so.1.23.2
│       └── ...
└── src/
    ├── main.cpp
    └── kws/kws.cpp
```
