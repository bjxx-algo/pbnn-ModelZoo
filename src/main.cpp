// #include <iostream>
// #include <iomanip>
// #include <string>

// #include "pb_sdk/pb_infer_api.h"
// #include "pb_sdk/qm_runtime.h"

// #include "yolov8s/preprocess.h"
// #include "yolov8s/postprocess.h"



// void yolov8s_det(std::string model_path){
//     std::string image_path = "data/inputc.jpg";
//     // std::string model_path = "model/yolov8s.pbnn";

//     //preprocess
//     std::cout << "Running preprocess..." << std::endl;
//     std::shared_ptr<yolov8sPreprocess> preprocessor = std::make_shared<yolov8sPreprocess>();
//     cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
//     torch::Tensor img_tensor = preprocessor->preprocess(img, 640);
//     std::cout << "Preprocess OK." << std::endl;

//     //infer
//     std::cout << "Running execute..." << std::endl;
//     int model_id = YOLOV8S;
//     ModelHandler model;
//     model.init(model_id, model_path);
//     CnnChatCompletions request;
//     CnnChatData part;
//     part.data_type = "float16";
//     part.data_shape = img_tensor.sizes().vec();
//     part.data.resize(img_tensor.nbytes());
//     std::memcpy(part.data.data(), img_tensor.data_ptr(), img_tensor.nbytes());
//     request.data_info.push_back(std::move(part));
//     request.case_name = "image";
//     model.input(request);
//     model.execute();
//     auto ret = model.output();
//     auto result = std::get<CnnChatCompletions>(ret);
//     std::cout << "Execute OK." << std::endl;

//     //postprocess
//     std::cout << "Running postprocess..." << std::endl;
//     bool draw_save_image = true;
//     DetectionResult det_result;
//     std::shared_ptr<YoloV8sPostprocess> postprocessor = std::make_shared<YoloV8sPostprocess>();
//     if (postprocessor == nullptr) {
//         std::cout << "YoloV8sDetector postprocessor is nullptr!";
//     }
//     postprocessor->Init();
//     postprocessor->postprocess(result.data_info[0].data.data(), img, det_result, draw_save_image);
//     std::cout << "Postprocess OK." << std::endl;

// }
// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <model_path>\n";
//         return -1;
//     }

//     std::string model_path = argv[1];
//     yolov8s_det(model_path);
//     return 0;
// }
#include <iostream>
#include <string>
#include <cstring>

// 各模型的入口函数声明
// int yolov8_main(int argc, char* argv[]);  // TODO: 待实现
int kws(int argc, char* argv[]);  // VAD+KWS 管道

// 帮助信息
const char* kModelZooHelp = R"(
Model Zoo - 预训练模型推理工具

用法: ./model-zoo <command> [options]

可用模型:
  yolov8s     YOLOv8s 目标检测
              ./model-zoo yolov8s <model_path> 
  vad-kws     语音活动检测 + 关键词识别管道
              ./model-zoo vad-kws \
                  --tokens=tokens.txt \
                  --encoder=encoder.pbnn \
                  --decoder=decoder.pbnn \
                  --joiner=joiner.pbnn \
                  --vad-model=vad.pbnn\
                  --keywords=keywords.txt \
                  audio.wav

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