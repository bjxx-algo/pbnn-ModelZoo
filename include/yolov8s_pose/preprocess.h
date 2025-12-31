/**
 * @file yolov8s_preprocess.h
 * @brief 图像预处理类，用于 YOLOv8 模型输入
 * @details 提供图像缩放、填充、BGR->RGB 转换、归一化等功能。
 */
#ifndef ADAS_MONOREPO_MODULES_PERCEPTION_CAMERA_DETECTION_GESTURE_DETECTOR_YOLOV8S_PREPROCESS_H
#define ADAS_MONOREPO_MODULES_PERCEPTION_CAMERA_DETECTION_GESTURE_DETECTOR_YOLOV8S_PREPROCESS_H

#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

// namespace apollo {
// namespace perception {
// namespace camera {

using namespace std;
namespace fs = std::filesystem;
class yolov8sPreprocess {
    public:
    /**
     * @brief 对输入图像进行预处理，返回 Torch 张量
     * @param image 输入的 OpenCV 图像（BGR 格式）
     * @param imgsz 模型输入尺寸（通常为 640）
     * @return torch::Tensor 形状为 [1, 3, H, W] 的 Float32 张量
     */
    torch::Tensor preprocess(const cv::Mat& image, int imgsz);
private:

    /**
    * @brief 对输入图像进行缩放与填充（letterbox）处理
    *
    * @param r 输出缩放比例（保持纵横比的缩放因子）
    * @param pad 输出填充值（左右、上下方向的像素填充量）
    * @param im 输入图像（BGR 格式）
    * @param new_shape 目标尺寸（默认 640x640）
    * @param color 填充颜色（默认灰色 114,114,114）
    * @param auto_ 是否根据 stride 自动调整填充（对齐到 stride 的倍数）
    * @param scaleFill 是否强制拉伸图像填满目标尺寸（不保持比例）
    * @param scaleup 是否允许放大图像（默认允许放大）
    * @param stride 网络步长（用于自动对齐，默认 32）
    * @return cv::Mat 返回经过缩放与填充后的图像
    */
    cv::Mat letterbox(float& r, cv::Point2f& pad,
                    const cv::Mat& im, cv::Size new_shape = cv::Size(640, 640),
                    cv::Scalar color = cv::Scalar(114, 114, 114),
                    bool auto_ = true, bool scaleFill = false, bool scaleup = true, int stride = 32);
};
// }  // namespace camera
// }  // namespace perception
// }  // namespace apollo

#endif // ADAS_MONOREPO_MODULES_PERCEPTION_CAMERA_DETECTION_GESTURE_DETECTOR_YOLOV8S_PREPROCESS_H