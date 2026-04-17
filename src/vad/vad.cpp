#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>

#include <torch/torch.h>


#include "preprocess.h"
#include "infer.h"
#include "postprocess.h"
#include "feature_config.h"


int vad(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " input.wav vad.onnx\n";
    return -1;
  }

  std::string wav_path = argv[1];
  std::string model_path = argv[2];

  try {
    auto total_start  = std::chrono::steady_clock::now();
    auto feat_start  = std::chrono::steady_clock::now();
    
    FeatureExtractor extractor(wav_path, SAMPLE_RATE);
    auto features = extractor.ExtractAll();
    auto feat_end  = std::chrono::steady_clock::now();
    
    auto infer_start   = std::chrono::steady_clock::now();
    TenVADInfer vad(model_path, /*threshold=*/0.5f,"vad");
    auto results = vad.Run(features);
    auto infer_end   = std::chrono::steady_clock::now();


    int total_frames = static_cast<int>(results.size());
    int speech_frames = 0;

    for (const auto& r : results) {
        if (r.vad == 1) {
            speech_frames++;
        }
    }

    double speech_ratio =
        total_frames > 0
            ? static_cast<double>(speech_frames) / total_frames
            : 0.0;

    auto total_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = total_end - total_start;
    std::chrono::duration<double> feat_time = feat_end - feat_start;
    std::chrono::duration<double> infer_time = infer_end - infer_start;
    // ============================
    // 3️⃣ 打印结果
    // ============================
    std::cout << "\n========== Performance Report ==========\n";
    std::cout << "Feature Extraction: "
              << feat_time.count() * 1000 << " ms\n";

    std::cout << "pbnn Inference:     "
              << infer_time.count() * 1000 << " ms\n";

    std::cout << "Total Pipeline3:     "
              << total_time.count() * 1000 << " ms\n";

    std::cout << "FPS:     "
              << (total_time.count() > 0 ? 1 / total_time.count() : 0) << " fps\n";

    std::cout << "========================================\n";


    std::cout << "[INFO] Post processing...\n";
    auto segments = ExtractSegmentsWithPostProcess(results);

    std::cout << "\nRunning completed:\n";
    std::cout << "Total frames: " << total_frames << "\n";
    std::cout << "Speech frames: " << speech_frames << "\n";
    std::cout << "Speech ratio: "
              << std::fixed << std::setprecision(2)
              << speech_ratio * 100 << "%\n";

    for (const auto& seg : segments) {
      std::cout
          << seg.start << " - "
          << seg.end << " : "
          << (seg.label ? "speech" : "silence")
          << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << std::endl;
    return -1;
  }
  return 0;
}
