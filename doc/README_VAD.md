# VAD — 语音活动检测 (Voice Activity Detection)

基于 TenVAD 的帧级语音活动检测，使用 40 维 Mel 滤波器组 + 能量特征，推理后端为 PBNN 加速引擎。

## 模型架构

| 组件 | 说明 |
|------|------|
| 前端特征 | 40-Mel FBank + 能量，预加重系数 0.97，上下文窗口 3 帧拼接 |
| 推理后端 | PBNN (`libten_vad_core.so`)，隐层维度 64，带 4 组隐藏状态 |
| 后处理 | 帧级 VAD 判决 → 语音段合并 → 段级后处理平滑 |

### 特征参数

| 参数 | 值 |
|------|------|
| 采样率 | 16000 Hz |
| FFT 大小 | 1024 |
| 窗长 | 768 samples (48 ms) |
| 帧移 | 256 samples (16 ms) |
| Mel 滤波器数 | 40 |
| 特征维度 | 41 (40 Mel + 1 能量) |
| 上下文窗口 | 3 帧 |

## 编译

### 环境要求

- CMake ≥ 3.16
- aarch64 交叉编译工具链（`aarch64-linux-gnu-g++`）
- libtorch aarch64 预编译库（CPU, ABI=1）
- PBNN SDK（`libpb_inference_engine.so`）

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
export LD_LIBRARY_PATH=/data/fuwj/pbnn-ModelZoo/lib/ten_vad/:/data/fuwj/pbnn-ModelZoo/lib/pb_sdk/:$LD_LIBRARY_PATH
```

### 2. 执行推理

```bash
./build/model-zoo vad input.wav vad.pbnn
```

### 参数说明

| 参数 | 位置 | 说明 |
|------|------|------|
| `input.wav` | 第 1 个参数 | 输入音频文件路径（16kHz 单声道 WAV） |
| `vad.pbnn` | 第 2 个参数 | VAD PBNN 模型文件路径 |

## 处理流程

```
WAV 文件
  │
  ▼
┌──────────────────────┐
│ 1. FeatureExtractor  │  加载音频 → 预加重 → 分帧 → FFT → Mel 滤波 → 归一化
│    (preprocess)      │
└──────────┬───────────┘
           │  torch::Tensor [N, 41]
           ▼
┌──────────────────────┐
│ 2. TenVADInfer       │  逐帧送入 PBNN 模型，输出 score → 阈值判决 (0.5)
│    (infer)           │
└──────────┬───────────┘
           │  vector<VadResult>
           ▼
┌──────────────────────┐
│ 3. PostProcess       │  帧级结果 → 合并相邻同类段 → 短段平滑
│    (postprocess)     │
└──────────┬───────────┘
           │  vector<Segment>
           ▼
       输出结果
```

## 输出格式

### 性能报告

```
========== Performance Report ==========
Feature Extraction: 18.8082 ms
pbnn Inference:     37.3155 ms
Total Pipeline3:     56.1325 ms
FPS:     17.815 fps
========================================
```

### VAD 统计

```
Running completed:
Total frames: 476
Speech frames: 329
Speech ratio: 69.12%
```

### 语音段输出

```
0.00 - 0.50 : silence
0.50 - 3.74 : speech
3.81 - 4.80 : silence
4.80 - 7.10 : speech
7.10 - 7.62 : silence
```

| 字段 | 说明 |
|------|------|
| `start` | 段起始时间（秒） |
| `end` | 段结束时间（秒） |
| `label` | `speech` 表示语音，`silence` 表示静音 |

## 核心数据结构

### VadResult（帧级结果）

```cpp
struct VadResult {
  int frame;        // 帧序号
  float timestamp;  // 时间戳 (秒)
  float score;      // VAD 置信度 [0, 1]
  int vad;          // 判决结果：1 = speech, 0 = silence
};
```

### Segment（段级结果）

```cpp
struct Segment {
  float start;  // 起始时间 (秒)
  float end;    // 结束时间 (秒)
  int label;    // 1 = speech, 0 = silence
};
```

## 目录结构

```
pbnn-ModelZoo/
├── CMakeLists.txt
├── toolchain-aarch64.cmake
├── include/
│   ├── ten_vad/
│   │   ├── feature_config.h    # 特征参数、均值/标准差
│   │   ├── preprocess.h        # FeatureExtractor 类
│   │   ├── infer.h             # TenVADInfer 类
│   │   └── postprocess.h       # 段合并与后处理
│   └── pb_sdk/
│       ├── pb_infer_api.h
│       └── qm_runtime.h
├── lib/
│   ├── ten_vad/
│   │   └── libten_vad_core.so
│   └── pb_sdk/
│       ├── libpb_inference_engine.so.3.1.1
│       ├── libonnxruntime.so.1.23.2
│       └── ...
└── src/
    ├── main.cpp
    └── vad/vad.cpp
```

## 依赖说明

| 依赖 | 用途 |
|------|------|
| libtorch (CPU, aarch64) | 特征提取中的 Tensor 运算、FFT、Mel 滤波 |
| PBNN SDK | 模型推理后端 |
| libten_vad_core | VAD 核心推理库 |
| libsndfile / libsamplerate | WAV 文件读取与重采样（通过 pb_sdk 间接依赖） |
