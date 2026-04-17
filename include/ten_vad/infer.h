#pragma once

// #include <onnxruntime_cxx_api.h>
#include <array>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "feature_config.h"
#include "pb_infer_api.h"
#include "qm_runtime.h"

struct VadResult {
  int frame;
  float timestamp;
  float score;
  int vad;
};

class TenVADInfer {
 public:
  TenVADInfer(const std::string& model_path, float threshold,std::string model_id);

  void Reset();

  std::vector<VadResult> Run(const torch::Tensor& features);

 private:
  ModelHandler model_;
  std::string model_id_;
  CnnChatCompletions request_;
  std::array<CnnChatData, 4> hidden_buf_;
  std::array<at::Half, CONTEXT_WINDOW_LEN * FEATURE_LEN> feature_buffer_;

  float threshold_ = 0.5f;
};
