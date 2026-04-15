#ifndef KWS_H_
#define KWS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <istream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include "pb_infer_api.h"
#include "qm_runtime.h"

// ===== Macros =====
#define SHERPA_ONNX_LOGE(...) do { \
  fprintf(stderr, "%s:%d ", __FILE__, __LINE__); \
  fprintf(stderr, ##__VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define SHERPA_ONNX_EXIT(code) exit(code)

#define SHERPA_ONNX_READ_META_DATA(dst, src_key) do { \
  auto value = LookupCustomModelMetaData(meta_data, src_key, allocator); \
  if (value.empty()) { SHERPA_ONNX_LOGE("'%s' missing", src_key); SHERPA_ONNX_EXIT(-1); } \
  dst = atoi(value.c_str()); } while(0)

#define SHERPA_ONNX_READ_META_DATA_VEC(dst, src_key) do { \
  auto value = LookupCustomModelMetaData(meta_data, src_key, allocator); \
  if (value.empty()) { SHERPA_ONNX_LOGE("'%s' missing", src_key); SHERPA_ONNX_EXIT(-1); } \
  if (!SplitStringToIntegers(value.c_str(), ",", true, &dst)) { \
    SHERPA_ONNX_LOGE("Bad '%s'", src_key); SHERPA_ONNX_EXIT(-1); } } while(0)

namespace sherpa_onnx {

// ===== No-op CHECK macros =====
class Voidifier {};
template <typename T> const Voidifier &operator<<(const Voidifier &v, T &&) { return v; }

} // namespace sherpa_onnx

#define SHERPA_ONNX_CHECK(x) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_LOG(x) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_EQ(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_NE(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_LT(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_LE(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_GT(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_CHECK_GE(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK(x) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DLOG(x) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_EQ(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_NE(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_LT(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_LE(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_GT(x, y) ::sherpa_onnx::Voidifier()
#define SHERPA_ONNX_DCHECK_GE(x, y) ::sherpa_onnx::Voidifier()

namespace sherpa_onnx {

// ===== Math =====
template <typename T> struct LogAdd;
template <> struct LogAdd<double> {
  double operator()(double x, double y) const {
    double d; if (x < y) { d = x - y; x = y; } else d = y - x;
    return d >= -36.04 ? x + log1p(exp(d)) : x;
  }
};
template <> struct LogAdd<float> {
  float operator()(float x, float y) const {
    float d; if (x < y) { d = x - y; x = y; } else d = y - x;
    return d >= -15.94f ? x + log1pf(expf(d)) : x;
  }
};
template <class T> void LogSoftmax(T *in, int32_t len) {
  T m = *std::max_element(in, in + len), s = 0;
  for (int32_t i = 0; i < len; i++) s += exp(in[i] - m);
  T off = m + log(s);
  for (int32_t i = 0; i < len; i++) in[i] -= off;
}
template <class T> void LogSoftmax(T *in, int32_t w, int32_t h) {
  for (int32_t i = 0; i != h; ++i, in += w) LogSoftmax(in, w);
}
template <class T> std::vector<int32_t> TopkIndex(const T *v, int32_t size, int32_t k) {
  k = std::min(k, size);
  // Heap-based selection: O(n log k) instead of O(n + k log n) with less allocation
  // For small k (e.g. 4) and large size (e.g. 1000+), this avoids allocating a
  // full index array and does less work overall.
  std::vector<int32_t> heap(k);
  for (int32_t i = 0; i < k; ++i) heap[i] = i;
  auto cmp = [v](int32_t a, int32_t b) { return v[a] > v[b]; };
  std::make_heap(heap.begin(), heap.end(), cmp);  // min-heap by value
  for (int32_t i = k; i < size; ++i) {
    if (v[i] > v[heap[0]]) {
      std::pop_heap(heap.begin(), heap.end(), cmp);
      heap.back() = i;
      std::push_heap(heap.begin(), heap.end(), cmp);
    }
  }
  std::sort_heap(heap.begin(), heap.end(), cmp);  // sort descending
  return heap;
}

// ===== Text utils =====
void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty, std::vector<std::string> *out);
template <class I>
bool SplitStringToIntegers(const std::string &full, const char *delim,
                           bool omit_empty, std::vector<I> *out) {
  if (full.empty()) { out->clear(); return true; }
  std::vector<std::string> split;
  SplitStringToVector(full, delim, omit_empty, &split);
  out->resize(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    char *end = nullptr;
    int64_t j = strtoll(split[i].c_str(), &end, 10);
    if (end == split[i].c_str() || *end != '\0') { out->clear(); return false; }
    I jI = static_cast<I>(j);
    if (static_cast<int64_t>(jI) != j) { out->clear(); return false; }
    (*out)[i] = jI;
  }
  return true;
}

// ===== File utils =====
bool FileExists(const std::string &filename);
std::vector<char> ReadFile(const std::string &filename);

// ===== Feature extraction =====
struct FeatureExtractorConfig {
  int32_t sampling_rate = 16000, feature_dim = 80;
  float low_freq = 20.0f, high_freq = -400.0f, dither = 0.0f;
  bool normalize_samples = true, snip_edges = false, remove_dc_offset = true;
  float frame_shift_ms = 10.0f, frame_length_ms = 25.0f, preemph_coeff = 0.97f;
  std::string window_type = "povey";
  bool round_to_power_of_two = true;
};

class FeatureExtractor {
 public:
  explicit FeatureExtractor(const FeatureExtractorConfig &config = {});
  ~FeatureExtractor();
  void AcceptWaveform(int32_t sr, const float *w, int32_t n) const;
  void InputFinished() const;
  int32_t NumFramesReady() const;
  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const;
  int32_t FeatureDim() const;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// ===== Wave reader =====
std::vector<float> ReadWave(const std::string &fname, int32_t *sr, bool *ok);
std::vector<float> ReadWave(std::istream &is, int32_t *sr, bool *ok);

// ===== ORT utils =====
void GetInputNames(Ort::Session *s, std::vector<std::string> *n, std::vector<const char *> *p);
void GetOutputNames(Ort::Session *s, std::vector<std::string> *n, std::vector<const char *> *p);
Ort::Value GetEncoderOutFrame(OrtAllocator *a, Ort::Value *enc_out, int32_t t);
Ort::Value Clone(OrtAllocator *a, const Ort::Value *v);
std::string LookupCustomModelMetaData(const Ort::ModelMetadata &m, const char *key, OrtAllocator *a);
Ort::Value Repeat(OrtAllocator *a, Ort::Value *enc_out, const std::vector<int32_t> &splits);

template <typename T = float>
void Fill(Ort::Value *t, T val) {
  auto n = t->GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementCount();
  std::fill(t->GetTensorMutableData<T>(), t->GetTensorMutableData<T>() + n, val);
}

template <typename T = float>
Ort::Value Cat(OrtAllocator *a, const std::vector<const Ort::Value *> &vals, int32_t dim);
template <typename T = float>
std::vector<Ort::Value> Unbind(OrtAllocator *a, const Ort::Value *val, int32_t dim);

// ===== Context graph =====
class ContextGraph;
using ContextGraphPtr = std::shared_ptr<ContextGraph>;

struct ContextState {
  int32_t token = -1;
  float token_score = 0, node_score = 0, output_score = 0;
  int32_t level = 0;
  float ac_threshold = 0;
  bool is_end = false;
  std::string phrase;
  std::unordered_map<int32_t, std::unique_ptr<ContextState>> next;
  const ContextState *fail = nullptr, *output = nullptr;
  ContextState() = default;
  ContextState(int32_t tok, float ts, float ns, float os, int32_t lv = 0,
               float act = 0, bool end = false, const std::string &ph = {})
      : token(tok), token_score(ts), node_score(ns), output_score(os),
        level(lv), ac_threshold(act), is_end(end), phrase(ph) {}
};

class ContextGraph {
 public:
  ContextGraph() = default;
  ContextGraph(const std::vector<std::vector<int32_t>> &ids, float score, float ac_thresh,
               const std::vector<float> &scores = {}, const std::vector<std::string> &phrases = {},
               const std::vector<float> &ac_thresholds = {});
  std::tuple<float, const ContextState *, const ContextState *>
  ForwardOneStep(const ContextState *state, int32_t token, bool strict = true) const;
  std::pair<bool, const ContextState *> IsMatched(const ContextState *state) const;
  std::pair<float, const ContextState *> Finalize(const ContextState *state) const;
  const ContextState *Root() const { return root_.get(); }
 private:
  float context_score_ = 0, ac_threshold_ = 0;
  std::unique_ptr<ContextState> root_;
  void Build(const std::vector<std::vector<int32_t>> &ids, const std::vector<float> &scores,
             const std::vector<std::string> &phrases, const std::vector<float> &thresholds) const;
  void FillFailOutput() const;
};

// ===== Hypothesis =====
struct Hypothesis {
  std::vector<int64_t> ys;
  std::vector<int32_t> timestamps;
  std::vector<float> ys_probs;
  double log_prob = 0, lm_log_prob = 0;
  const ContextState *context_state = nullptr;
  int32_t num_trailing_blanks = 0;
  Hypothesis() = default;
  Hypothesis(const std::vector<int64_t> &ys, double lp, const ContextState *cs = nullptr)
      : ys(ys), log_prob(lp), context_state(cs) {}
  double TotalLogProb() const { return log_prob + lm_log_prob; }
  std::string Key() const {
    std::ostringstream os;
    std::string sep;
    for (auto i : ys) { os << sep << i; sep = "-"; }
    return os.str();
  }
};

class Hypotheses {
 public:
  Hypotheses() = default;
  explicit Hypotheses(std::vector<Hypothesis> hyps) {
    for (auto &h : hyps) hyps_[h.Key()] = std::move(h);
  }
  void Add(Hypothesis hyp);
  Hypothesis GetMostProbable(bool norm) const;
  std::vector<Hypothesis> GetTopK(int32_t k, bool norm) const;
  int32_t Size() const { return hyps_.size(); }
  auto begin() const { return hyps_.begin(); }
  auto end() const { return hyps_.end(); }
  auto begin() { return hyps_.begin(); }
  auto end() { return hyps_.end(); }
  std::vector<Hypothesis> Vec() const {
    std::vector<Hypothesis> r;
    r.reserve(hyps_.size());
    for (auto &p : hyps_) r.push_back(p.second);
    return r;
  }
 private:
  std::unordered_map<std::string, Hypothesis> hyps_;
};

const std::vector<int32_t> GetHypsRowSplits(const std::vector<Hypotheses> &hyps);

// ===== TransducerKeywordResult =====
struct TransducerKeywordResult {
  int32_t frame_offset = 0;
  std::vector<int64_t> tokens;
  std::string keyword;
  int32_t num_trailing_blanks = 0;
  std::vector<int32_t> timestamps;
  Hypotheses hyps;
};

// ===== Model =====
class OnlineStream;

class OnlineZipformer2TransducerModel {
 public:
  OnlineZipformer2TransducerModel(const std::string &encoder_path,
      const std::string &decoder_path, const std::string &joiner_path, int32_t threads = 1);
  std::vector<Ort::Value> StackStates(const std::vector<std::vector<Ort::Value>> &states) const;
  std::vector<std::vector<Ort::Value>> UnStackStates(const std::vector<Ort::Value> &states) const;
  std::vector<Ort::Value> GetEncoderInitStates();
  void SetFeatureDim(int32_t d) { feature_dim_ = d; }
  std::pair<Ort::Value, std::vector<Ort::Value>> RunEncoder(
      Ort::Value features, std::vector<Ort::Value> states, Ort::Value processed_frames);

  void assignInput(Ort::Value& decoder_input, Ort::Value& encoder_out);
  void InitDecoderandJoiner(std::string pbnn_path);
  Ort::Value RunDecoderandJoiner(Ort::Value decoder_input, Ort::Value encoder_out);    

  Ort::Value RunDecoder(Ort::Value input);
  Ort::Value RunJoiner(Ort::Value enc, Ort::Value dec);
  Ort::Value BuildDecoderInput(const std::vector<Hypothesis> &hyps);
  int32_t ContextSize() const { return context_size_; }
  int32_t ChunkSize() const { return T_; }
  int32_t ChunkShift() const { return decode_chunk_len_; }
  int32_t VocabSize() const { return vocab_size_; }
  OrtAllocator *Allocator() { return allocator_; }
 private:
  void InitEncoder(void *data, size_t len);
  void InitDecoder(void *data, size_t len);
  void InitJoiner(void *data, size_t len);
  CnnChatCompletions request_;
  CnnChatData part_[2];
  ModelHandler pbnnmodel;
  Ort::Env env_;
  Ort::SessionOptions enc_opts_, dec_opts_, join_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> enc_sess_, dec_sess_, join_sess_;
  std::vector<std::string> enc_in_, enc_out_, dec_in_, dec_out_, join_in_, join_out_;
  std::vector<const char *> enc_in_p_, enc_out_p_, dec_in_p_, dec_out_p_, join_in_p_, join_out_p_;


  std::vector<int32_t> encoder_dims_, query_head_dims_, value_head_dims_;
  std::vector<int32_t> num_heads_, num_encoder_layers_, cnn_module_kernels_, left_context_len_;
  int32_t T_ = 0, decode_chunk_len_ = 0, context_size_ = 0, vocab_size_ = 0, feature_dim_ = 80;
};

// ===== OnlineStream =====
class OnlineStream {
 public:
  explicit OnlineStream(const FeatureExtractorConfig &config = {},
                        ContextGraphPtr cg = nullptr);
  ~OnlineStream();
  void AcceptWaveform(int32_t sr, const float *w, int32_t n) const;
  void InputFinished() const;
  int32_t NumFramesReady() const;
  std::vector<float> GetFrames(int32_t fi, int32_t n) const;
  void Reset();
  int32_t FeatureDim() const;
  int32_t &GetNumProcessedFrames();
  int32_t GetNumFramesSinceStart() const;
  void SetKeywordResult(const TransducerKeywordResult &r);
  TransducerKeywordResult &GetKeywordResult(bool remove_dup = false);
  void SetStates(std::vector<Ort::Value> s);
  std::vector<Ort::Value> &GetStates();
  const ContextGraphPtr &GetContextGraph() const;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// ===== SymbolTable =====
class SymbolTable {
 public:
  SymbolTable() = default;
  explicit SymbolTable(const std::string &filename);
  const std::string operator[](int32_t id) const;
  int32_t operator[](const std::string &sym) const;
  bool Contains(const std::string &sym) const { return sym2id_.count(sym) != 0; }
 private:
  void Init(std::istream &is);
  std::unordered_map<std::string, int32_t> sym2id_;
  std::unordered_map<int32_t, std::string> id2sym_;
  bool bpe_byte_fallback_ = false;
  int32_t id_for_0x00_ = 0;
};

// ===== TransducerKeywordDecoder =====
class TransducerKeywordDecoder {
 public:
  TransducerKeywordDecoder(OnlineZipformer2TransducerModel *m, int32_t max_paths,
                           int32_t trailing_blanks, int32_t unk_id)
      : model_(m), max_paths_(max_paths), trailing_blanks_(trailing_blanks), unk_id_(unk_id) {}
  TransducerKeywordResult GetEmptyResult() const;
  void Decode(Ort::Value enc_out, OnlineStream **ss,
              std::vector<TransducerKeywordResult> *result);
 private:
  OnlineZipformer2TransducerModel *model_;
  int32_t max_paths_, trailing_blanks_, unk_id_;
};

bool EncodeKeywords(std::istream &is, const SymbolTable &st,
                    std::vector<std::vector<int32_t>> *ids, std::vector<std::string> *kws,
                    std::vector<float> *scores, std::vector<float> *thresholds);

// ===== KeywordResult =====
struct KeywordResult {
  std::string keyword;
  std::vector<std::string> tokens;
  std::vector<float> timestamps;
  float start_time = 0;
  std::string AsJsonString() const;
};

// ===== KeywordSpotter =====
struct KwsConfig {
  std::string encoder, decoder, joiner, tokens, keywords_file;
  FeatureExtractorConfig feat;
  int32_t num_threads = 1, max_active_paths = 4, num_trailing_blanks = 1;
  float keywords_score = 1.0f, keywords_threshold = 0.25f;
};

class KeywordSpotter {
 public:
  explicit KeywordSpotter(const KwsConfig &config);
  ~KeywordSpotter();
  std::unique_ptr<OnlineStream> CreateStream() const;
  bool IsReady(OnlineStream *s) const;
  void Reset(OnlineStream *s) const;
  void DecodeStream(OnlineStream *s) const;
  void DecodeStreams(OnlineStream **ss, int32_t n) const;
  KeywordResult GetResult(OnlineStream *s) const;
 private:
  void InitOnlineStream(OnlineStream *s) const;
  KwsConfig config_;
  std::unique_ptr<OnlineZipformer2TransducerModel> model_;
  std::unique_ptr<TransducerKeywordDecoder> decoder_;
  SymbolTable sym_;
  int32_t unk_id_ = -1;
  std::vector<std::vector<int32_t>> keywords_id_;
  std::vector<std::string> keywords_;
  std::vector<float> boost_scores_, thresholds_;
  ContextGraphPtr keywords_graph_;
};

} // namespace sherpa_onnx

#endif // KWS_H_
