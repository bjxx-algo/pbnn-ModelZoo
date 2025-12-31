#ifndef ADAS_MONOREPO_MODULES_PERCEPTION_CAMERA_DETECTION_GESTURE_DETECTOR_YOLOV8S_POSTPROCESS_H
#define ADAS_MONOREPO_MODULES_PERCEPTION_CAMERA_DETECTION_GESTURE_DETECTOR_YOLOV8S_POSTPROCESS_H
/**
 * @file yolov8s_postprocess.h
 * @brief 图像后处理类，用于 YOLOv8 模型输出
 * @details 提供nms等功能。
 */
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <torch/torch.h>
#include <torch/script.h>

#include "common.h"

#include <unistd.h>

// namespace camera {


using namespace std;
namespace fs = std::filesystem;

/**
 * @brief YOLOv8s 后处理类
 * @details 负责模型输出的解析、结果绘制及保存。
 */
class YoloV8sPostprocess {
public:
  /**
  * @brief 构造函数，初始化后处理对象
  */
  YoloV8sPostprocess();

  /**
   * @brief 析构函数，释放资源
   */
  ~YoloV8sPostprocess();

  /**
   * @brief 初始化后处理模块
   * @return 初始化是否成功
   */
  bool Init();

  /**
   * @brief 反初始化，释放资源
   * @return 释放是否成功
   */

  bool DeInit();

  /**
   * @brief 保存张量为二进制文件
   * @param tensor 要保存的张量
   * @param filename 输出文件路径
   */
  void save_tensor_binary(
    const torch::Tensor& tensor, const std::string& filename);

/**
 * @brief YOLOv8 模型的后处理函数
 * @details
 * 该函数用于对模型推理输出结果进行后处理，包括解码检测框、筛选目标、绘制结果等操作。
 * 可选地将结果绘制到原图上并保存。
 *
 * @param[out] out_data       模型推理输出数据结构体，包含输出 tensor 数据及长度信息
 * @param[in,out] cv_image    输入图像（BGR 格式），当 draw_save_image 为 true 时，
 *                            函数会在该图像上绘制检测框
 * @param[out] det_result     检测结果结构体，包含每个目标的类别、置信度和边界框坐标
 * @param[in] draw_save_image 是否在图像上绘制检测结果并保存，默认为 false
 *
 * @return true  表示后处理成功  
 * @return false 表示后处理失败（如数据为空或解析错误）
 */
  bool postprocess(
    uint8_t* out_data, cv::Mat& cv_image,DetectionResult &det_result, bool draw_save_image = false);

private:
/**
 * @brief 初始化 YOLOv8s 模型
 * @param model_path 模型文件路径
 * @param thread_nums 推理使用的线程数
 * @return 初始化是否成功
 */
  // bool InitModel(const std::string& model_path, const std::uint32_t thread_nums);

/**
 * @brief 绘制检测到的关键点与连接线
 * @param image 输入/输出图像
 * @param boxes 检测到的目标框
 * @param keypoints 关键点坐标
 * @param connections 关键点之间的连接关系（默认骨架连接）
 * @param line_color 连接线颜色（默认绿色，BGR: (60,179,113)）
 * @param point_color 关键点颜色（默认红色，BGR: (0,0,255)）
 * @param offset 坐标偏移量（用于平移关键点位置）
 * @param show_idx 是否在关键点旁显示索引号
 */
  void plot_keypoints(cv::Mat& image,
                      const std::vector<cv::Rect>& boxes,
                      const std::vector<cv::Point2f>& keypoints,
                      const std::vector<std::pair<int, int>>& connections = {
                          {0,1}, {1,2}, {2,3}, {3,4}, {0,5}, {5,6}, {6,7}, {7,8},
                          {5,9}, {9,10}, {10,11}, {11,12}, {9,13}, {13,14}, {14,15}, {15,16},
                          {13,17}, {0,17}, {17,18}, {18,19}, {19,20}
                      },
                      const cv::Scalar& line_color = cv::Scalar(113, 179, 60),   // BGR: (60,179,113)
                      const cv::Scalar& point_color = cv::Scalar(0, 0, 255),     // BGR: (255,0,0)
                      const cv::Point2f& offset = cv::Point2f(0.0f, 0.0f),
                      bool show_idx = false);

/**
 * @brief 将坐标限制在图像边界内
 * @param coords 输入坐标张量
 * @param shape 图像尺寸（宽和高）
 * @return 限制在边界内的坐标张量
 */                
  torch::Tensor clip_coords(torch::Tensor coords, const cv::Size& shape);

/**
 * @brief 计算两个矩形框的 IoU（Intersection over Union）
 * @param box1 第一个矩形框张量 [N, 4]
 * @param box2 第二个矩形框张量 [M, 4]
 * @param eps 防止除零的极小值（默认 1e-7）
 * @return IoU 值张量 [N, M]
 */                  
  torch::Tensor box_iou(torch::Tensor box1, torch::Tensor box2, float eps = 1e-7);

  /**
 * @brief 计算旋转框的协方差矩阵
 * @param boxes 旋转框张量 [N, 5]（x, y, w, h, θ）
 * @return 返回三个张量：均值、协方差矩阵、旋转矩阵
 */                  
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_covariance_matrix(const torch::Tensor& boxes);
    
  /**
 * @brief 批量计算旋转框的 ProbIoU（概率 IoU）
 * @param obb1 第一组旋转框张量 [N, 5]
 * @param obb2 第二组旋转框张量 [M, 5]
 * @param eps 防止数值不稳定的极小值（默认 1e-7）
 * @return ProbIoU 值张量 [N, M]
 */
  torch::Tensor batch_probiou(const torch::Tensor& obb1, const torch::Tensor& obb2, float eps = 1e-7);
    
  /**
 * @brief 对旋转矩形框执行非极大值抑制（Rotated NMS）
 * @param boxes 旋转矩形框张量 [N, 5]（x, y, w, h, θ）
 * @param scores 置信度分数张量 [N]
 * @param threshold IoU 阈值（默认 0.45）
 * @return 保留框的索引张量
 */
  torch::Tensor nms_rotated(torch::Tensor boxes, torch::Tensor scores, float threshold = 0.45);
  
  /**
 * @brief 将矩形框格式从 (x, y, w, h) 转换为 (x1, y1, x2, y2)
 * @param x 输入框张量 [N, 4]
 * @return 转换后的框张量 [N, 4]
 */
  torch::Tensor xywh2xyxy(const torch::Tensor& x);
 
  /**
 * @brief 对普通矩形框执行非极大值抑制（NMS）
 * @param boxes 矩形框张量 [N, 4]
 * @param scores 置信度分数张量 [N]
 * @param iou_threshold IoU 阈值
 * @return 保留框的索引张量
 */
  torch::Tensor nms(const torch::Tensor& boxes, const torch::Tensor& scores, double iou_threshold);
    
  /**
 * @brief 执行批量非极大值抑制（NMS），用于过滤 YOLO 检测结果
 * @param prediction 模型预测结果张量，形状通常为 [batch, num_boxes, num_classes + 5]
 * @param conf_thres 置信度阈值（默认 0.25），低于该值的检测将被忽略
 * @param iou_thres IoU 阈值（默认 0.45），用于 NMS 判定框重叠
 * @param classes 指定保留的类别索引（默认空，即保留所有类别）
 * @param agnostic 是否类别无关（true 表示忽略类别进行 NMS）
 * @param multi_label 是否支持多标签检测（即一个框可属于多个类别）
 * @param max_det 每张图像最多保留的检测框数量（默认 300）
 * @param nc 类别数量（num_classes，默认 0）
 * @param max_time_img 单张图像最大处理时间（秒），防止过长推理
 * @param max_nms NMS 阶段最多考虑的候选框数量（默认 30000）
 * @param max_wh 限制坐标的最大宽高值（默认 7680，用于偏移不同类别的框）
 * @param in_place 是否原地修改 prediction 张量（节省显存）
 * @param rotated 是否为旋转框 NMS（true 表示处理旋转目标）
 * @return 每张图像对应的检测结果张量列表（每个张量形状为 [num_det, 6]：x1, y1, x2, y2, conf, cls）
 */
  std::vector<torch::Tensor> non_max_suppression(
      torch::Tensor prediction,
      float conf_thres = 0.25,
      float iou_thres = 0.45,
      std::vector<int> classes = {},
      bool agnostic = false,
      bool multi_label = false,
      int max_det = 300,
      int nc = 0,
      float max_time_img = 0.05,
      int max_nms = 30000,
      int max_wh = 7680,
      bool in_place = true,
      bool rotated = false
  );
  /**
 * @brief 将预测框裁剪到图像范围内，防止越界
 * @param boxes 输入的边界框张量 [N, 4]
 * @param shape 图像尺寸（cv::Size，包含宽和高）
 * @return 裁剪后的边界框张量 [N, 4]
 */
  torch::Tensor clip_boxes(const torch::Tensor& boxes, const cv::Size& shape);
  /**
 * @brief 根据图像缩放比例，将预测框从模型输入尺寸映射回原始图像尺寸
 * @param img1_shape 模型输入图像尺寸 {height, width}
 * @param boxes 边界框张量 [N, 4] 或 [N, >=4]
 * @param img0_shape 原始图像尺寸 {height, width}
 * @param ratio_pad 可选的缩放比例与填充参数 (ratio, pad)，若提供则使用精确映射
 * @param padding 是否考虑 letterbox 填充（默认 true）
 * @param xywh 输入框是否为 (x, y, w, h) 格式（默认 false 表示 (x1, y1, x2, y2)）
 * @return 映射回原图坐标系的边界框张量 [N, 4]
 */
  torch::Tensor scale_boxes(
      const std::vector<int>& img1_shape,    // {height, width}
      torch::Tensor boxes,                   // [N, 4] or [N, >=4]
      const std::vector<int>& img0_shape,    // {height, width}
      std::optional<std::pair<torch::Tensor, std::vector<float>>> ratio_pad = std::nullopt,
      bool padding = true,
      bool xywh = false
  );
  
  /**
 * @brief 将关键点或坐标从一种图像尺寸映射到另一种图像尺寸
 * @param from_shape 源图像尺寸 {height, width}
 * @param coords 坐标张量 [N, 2] 或 [N, >=2]
 * @param to_img 目标图像，用于获取目标分辨率
 * @return 映射到目标图像坐标系的坐标张量
 */
  torch::Tensor scale_coords(const std::vector<int>& from_shape, torch::Tensor coords, const cv::Mat& to_img);
  
  /**
 * @brief 根据索引返回调色板中的颜色，用于可视化不同类别
 * @param idx 索引
 * @return BGR 格式的颜色 (cv::Scalar)
 */
  cv::Scalar colors(int idx);
  
  /**
 * @brief 在图像上绘制边界框和标签
 * @param img 输入图像
 * @param box 边界框张量 [x1, y1, x2, y2]
 * @param label 标签文本
 * @param color 边界框颜色
 */
  void draw_box_label(cv::Mat& img, const torch::Tensor& box, const std::string& label, const cv::Scalar& color);
  
  /**
 * @brief 在图像上绘制关键点
 * @param img 输入图像
 * @param kpts 关键点张量 [N, 2]，每行为 (x, y)
 */
  void plot_keypoints(cv::Mat& img, const torch::Tensor& kpts);
  
  /**
 * @brief 将原始指针数据转换为 PyTorch 张量
 * @param ptr 数据指针
 * @param dtype 数据类型
 * @param sizes 张量形状
 * @return 转换后的张量
 */
  torch::Tensor ptr_to_tensor(uint8_t* ptr, torch::Dtype dtype, std::vector<int64_t> shape);
  
  void saveImageWithTimestamp(const cv::Mat& image, const std::string& dir = "./results");

  int postProcessAnnotate(
      uint8_t* Reshape_output_0,DetectionResult &det_result,
      cv::Mat& image,
      int imgsz,
      float conf_thres,
      float iou_thres,
      int max_det, bool draw_save_image = false
  );
  void compare_tensors(const torch::Tensor& a, const torch::Tensor& b);
  template<typename T>
  bool loadBinaryFile(const std::string& filename, std::vector<T>& data);
  at::Tensor load_bin_half(const std::string& det_out_path, const std::vector<int64_t>& new_shape) ;

private:
    // Ort::Env env_;
    // std::shared_ptr<Ort::Session> session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::map<int, std::string> names_;
    std::string model_path_ ;
    // std::uint32_t onnx_thread_nums_{8};
    // std::uint32_t onnx_graph_optimization_level_{2};
    std::vector<int32_t> classes_{18, 20, 21, 22};

};
// }  // namespace camera

#endif // ADAS_MONOREPO_MODULES_PERCEPTION_CAMERA_DETECTION_GESTURE_DETECTOR_YOLOV8S_POSTPROCESS_H