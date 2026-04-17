#pragma once
#include <vector>
#include <torch/torch.h>


struct FeatureExtractor {
  FeatureExtractor(const std::string& wav_path,int expected_sr);

  torch::Tensor ExtractFrame(const torch::Tensor& frame);
  torch::Tensor ExtractAll();
  // torch::Tensor ExtractAll(const torch::Tensor& audio);

 private:
  torch::Tensor mel_filters_;
  torch::Tensor window_;
  float pre_emphasis_prev_;
  std::string wav_path_;
  int expected_sr_;
  torch::Tensor PreEmphasis(const torch::Tensor& x);
  torch::Tensor LoadWav(const std::string& path, int expected_sr) ;
};
