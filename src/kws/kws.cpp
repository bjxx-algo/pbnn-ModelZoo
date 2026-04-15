#include "kws.h"
#include <chrono>

int kws(int argc, char *argv[]) {
  sherpa_onnx::KwsConfig config;
  std::string wav;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) != "--") { wav = arg; continue; }
    auto eq = arg.find('=');
    if (eq == std::string::npos) continue;
    std::string key = arg.substr(2, eq - 2), val = arg.substr(eq + 1);
    if (key == "encoder") config.encoder = val;
    else if (key == "decoder") config.decoder = val;
    else if (key == "joiner") config.joiner = val;
    else if (key == "tokens") config.tokens = val;
    else if (key == "keywords-file") config.keywords_file = val;
    else if (key == "max-active-paths") config.max_active_paths = std::stoi(val);
    else if (key == "num-trailing-blanks") config.num_trailing_blanks = std::stoi(val);
    else if (key == "keywords-score") config.keywords_score = std::stof(val);
    else if (key == "keywords-threshold") config.keywords_threshold = std::stof(val);
    else if (key == "sample-rate") config.feat.sampling_rate = std::stoi(val);
    else if (key == "feat-dim") config.feat.feature_dim = std::stoi(val);
    else if (key == "num-threads") config.num_threads = std::stoi(val);
  }

  if (config.encoder.empty() || config.decoder.empty() || config.joiner.empty() ||
      config.tokens.empty() || config.keywords_file.empty() || wav.empty()) {
    fprintf(stderr, "Usage: %s --encoder=... --decoder=... --joiner=... "
            "--tokens=... --keywords-file=... input.wav\n", argv[0]);
    return 1;
  }

  sherpa_onnx::KeywordSpotter spotter(config);
  int32_t sr = -1;
  bool ok = false;
  auto samples = sherpa_onnx::ReadWave(wav, &sr, &ok);
  if (!ok) { fprintf(stderr, "Failed to read '%s'\n", wav.c_str()); return 1; }

  auto begin = std::chrono::steady_clock::now();
  auto s = spotter.CreateStream();

  s->AcceptWaveform(sr, samples.data(), samples.size());
  std::vector<float> tail(static_cast<int>( 0.6 * sr));
  s->AcceptWaveform(sr, tail.data(), tail.size());
  
  s->InputFinished();

  while (spotter.IsReady(s.get())) {
    spotter.DecodeStream(s.get());
    auto r = spotter.GetResult(s.get());
    if (!r.keyword.empty()) {
      spotter.Reset(s.get());
      fprintf(stderr, "%s\n%s\n\n", wav.c_str(), r.AsJsonString().c_str());
    }
  }

  auto end = std::chrono::steady_clock::now();
  float dur = samples.size() / (float)sr;
  float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.f;

  fprintf(stderr, "Duration: %.3fs | Elapsed: %.3fs | RTF: %.3f\n", dur, elapsed, elapsed / dur);
  return 0;
}
