#ifndef YOLOV8S_COMMON_H_
#define YOLOV8S_COMMON_H_

#include <cstdint>

// namespace camera {

/**
 * @brief 输出数据结构，用于封装模型输出的多条数据
 */
typedef struct {
    uint8_t **data;
    uint32_t *data_len;
    uint32_t data_count;
} OUT_DATA;


struct BBox {
  int x1;
  int y1;
  int x2;
  int y2;
};

struct DetectionResult {
  std::string label;
  float conf;
  std::string type;  // "face", "hand" 等
  BBox box;
};
// }  // namespace camera



#endif  // YOLOV8S_COMMON_H_
