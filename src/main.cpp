#include <iostream>
#include <string>
#include <cstring>

// 各模型的入口函数声明
// int yolov8_main(int argc, char* argv[]);  // TODO: 待实现
int kws(int argc, char* argv[]);  // KWS 管道
int vad(int argc, char** argv) ;  // 语音活动检测
// 帮助信息
const char* kModelZooHelp = R"(
Model Zoo - 预训练模型推理工具

用法: ./model-zoo <command> [options]

可用模型:
  yolov8s     YOLOv8s 目标检测
              ./model-zoo yolov8s <model_path> 
  kws     关键词识别管道
            ./model-zoo kws \
            --tokens=tokens.txt \
            --encoder=encoder.pbnn \
            --decoder=decoder.pbnn \
            --joiner=joiner.pbnn \
            --vad-model=vad.pbnn\
            --keywords=keywords.txt \
            audio.wav
  vad     语音活动检测
              ./model-zoo vad <audio_path> <model_path> 
  help        显示此帮助信息

示例:
  # YOLOv8 检测
  ./model-zoo yolov8s yolov8s.onnx photo.jpg

  # VAD+KWS 语音处理
  ./model-zoo vad-kws --tokens=tokens.txt --encoder=enc.onnx \
      --decoder=dec.onnx --joiner=joiner.onnx --vad-model=vad.onnx \
      --keywords=keywords.txt recording.wav
)";

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << kModelZooHelp;
        return -1;
    }

    std::string command = argv[1];

    // 移除 command 本身，调整 argc/argv 传递给子模块
    int sub_argc = argc - 1;
    char** sub_argv = argv + 1;
    sub_argv[0] = argv[0];  // 保持程序名

    if (command == "yolov8s" || command == "yolov8") {
        // return yolov8_main(sub_argc, sub_argv);  // TODO: 待实现
        std::cerr << "yolov8s 暂未启用\n";
        return -1;
    }
    else if (command == "kws") {
        return kws(sub_argc, sub_argv);
    }
    else if (command == "vad") {
        return vad(sub_argc, sub_argv);
    }
    else if (command == "help" || command == "-h" || command == "--help") {
        std::cout << kModelZooHelp;
        return 0;
    }
    else {
        std::cerr << "未知命令: " << command << "\n\n";
        std::cerr << kModelZooHelp;
        return -1;
    }
}