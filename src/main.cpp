#include <iostream>
#include <iomanip>
#include <string>

#include "pb_sdk/pb_infer_api.h"
#include "pb_sdk/qm_runtime.h"

#include "yolov8s/preprocess.h"
#include "yolov8s/postprocess.h"



void yolov8s_det(std::string model_path){
    std::string image_path = "data/inputc.jpg";
    // std::string model_path = "model/yolov8s.pbnn";

    //preprocess
    std::cout << "Running preprocess..." << std::endl;
    yolov8sPreprocess pre = yolov8sPreprocess();
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    torch::Tensor img_tensor = pre.preprocess(img, 640);
    std::cout << "Preprocess OK." << std::endl;

    //infer
    std::cout << "Running execute..." << std::endl;
    int model_id = YOLOV8S;
    ModelHandler model;
    model.init(model_id, model_path);
    CnnChatCompletions request;
    CnnChatData part;
    part.data_type = "float16";
    part.data_shape = img_tensor.sizes().vec();
    part.data.resize(img_tensor.nbytes());
    std::memcpy(part.data.data(), img_tensor.data_ptr(), img_tensor.nbytes());
    request.data_info.push_back(std::move(part));
    request.case_name = "image";
    model.input(request);
    model.execute();
    auto ret = model.output();
    auto result = std::get<CnnChatCompletions>(ret);
    std::cout << "execute OK." << std::endl;


    //postprocess
    std::cout << "Running postprocess..." << std::endl;
    bool draw_save_image = true;
    DetectionResult det_result;
    std::shared_ptr<YoloV8sPostprocess> postprocessor = std::make_shared<YoloV8sPostprocess>();
    if (postprocessor == nullptr) {
        std::cout << "YoloV8sDetector postprocessor is nullptr!";
    }
    postprocessor->Init();
    postprocessor->postprocess(result.data_info[0].data.data(), img, det_result, draw_save_image);
    std::cout << "Postprocess OK." << std::endl;

}
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>\n";
        return -1;
    }

    std::string model_path = argv[1];
    yolov8s_det(model_path);
    return 0;
}