#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

enum class UserRequestType {
    INIT_MODEL = 0,
    TERMINATE_MODEL,
    CHAT_COMPLETIONS,
    CHAT_COMPLETIONS_STREAM,
    ABORT_CHAT,
    LOAD_KV_CACHE,
    SAVE_KV_CACHE,
    START_TRACE,
    STOP_TRACE,

    CNN_CHAT_COMPLETIONS,
    GET_NPU_STATUS,
    UNLOAD_MODEL,
};

enum ModelType {
    MINICPM = 1,
    DEEPSEEK = 2,
    PHI4 = 3,
    MINICPMV1B = 4,
    QWEN = 5,
    PALIGEMMA = 6,
    PALIGEMMA_V1_1 = 7,
    QWEN_2_5VL_7B = 8,
    QWEN_2_5VL_3B = 9,
    QWEN_2_5VL_7B_DA04 = 10,
    QWEN_2_5OMNI_7B_DA04 = 11,
    INTERNVL3_8B = 12,
    QWEN_2_5VL_7B_EAGLE3_DA04 = 13,
    FIRST_CNN_MODEL= 1000,
    RESNET50 = FIRST_CNN_MODEL,
    REPVGG,
    YOLOV8S
};

enum SamplerType {
    GREEDY = 1,
    TEMPERATURE = 2
};

enum ErrCode {
    FAILED = -1,
    SUCCESS = 0,
    TIMEOUT,
    INVALID_MODEL ,
    INIT_VOCAB_ERROR,
    INIT_EMBEDDING_TABLE_ERROR,
    TOKENIZE_ERROR,
    DETOKENIZE_ERROR,
    PRE_ATTN_PREFILL_ERROR,
    PRE_ATTN_DECODE_ERROR,
    ATTN_ERROR,
    POST_ATTN_PREFILL_ERROR,
    POST_ATTN_DECODE_ERROR,
    LM_HEAD_ERROR,
    INIT_PLATFORM_ERROR,
    MODEL_MISMATCH_PLATFORM,
};

struct FunctionCall {
    std::string name;
    std::string arguments;
};

struct ToolCall {
    std::string id;
    std::string type;
    FunctionCall function;
};

struct ReqEntity{
    virtual ~ReqEntity()=default;
};

struct ContentPart {
    std::string type;
    std::string text;
    struct {
        std::string url;
    } image_url;
    struct {
        std::string data;
    } input_audio;
    std::vector<std::string> video;
    std::optional<std::vector<uint16_t>> pixel_data; // image/video pixel data in float16 format
};

struct Message {
    std::string role;
    std::vector<ContentPart> content;
    std::vector<ToolCall> tool_calls;
};

struct ResponseFormat {
    std::string type;
    struct {
        std::string schema;
    } json_schema;
    std::string regex;
};

struct ChatCompletionsRequest:public ReqEntity {
    // OpenAI Compatible API parameters
    std::vector<Message> messages;
    std::vector<std::string> tools;
    std::string model;
    std::optional<int> max_completion_tokens = std::nullopt;
    std::optional<ResponseFormat> response_format = std::nullopt;
    std::optional<int> seed = std::nullopt;
    bool stream = false;
    float temperature = 1.0f;
    float top_p = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;

    // Additional parameters
    int top_k = 50;
    int meta_top_k = 100;
    bool ignore_eos = false;
    bool no_cpu_npu_parallel = false;
    bool no_prefix_cache = false;
    float repetition_penalty = 1.0f;

    std::vector<std::vector<float>> imu;
    float fps = 1;
    std::string data_type = "image";
};

struct Usage {
    int completion_tokens;
    int prompt_tokens;
    int total_tokens;
};

struct Metric {
    double prefill_time;
    double prefill_speed;
    double decode_time;
    double decode_speed;
    double ve_time;
    double preprocess_time;
    double average_accept_length;
    int max_accept_length;
    int min_accept_length;
    double iterations_speed;
};

struct ChatCompletionChoice {
    std::string finish_reason;
    int index;
    struct {
        std::string role;
        std::optional<std::string> content;
        std::vector<ToolCall> tool_calls;
    } message;
};

// "object": "chat.completion"
struct ChatCompletionObject {
    std::vector<ChatCompletionChoice> choices;
    time_t created;
    std::string model;
    std::string system_fingerprint;
    Usage usage;
    std::string id;

    // Additional fields
    Metric metric;
};

struct ChatCompletionChunkChoiceDelta {
    std::optional<std::string> content;
    std::optional<std::string> role;
};

struct ChatCompletionChunkChoice {
    std::optional<std::string> finish_reason;
    int index;
    struct {
        // According to the behavior of official OpenAI API, at most one of
        // these fields will be present in a chunk
        std::optional<std::string> content;
        std::optional<std::string> role;
    } delta;
};

// "object": "chat.completion.chunk"
struct ChatCompletionChunkObject {
    std::vector<ChatCompletionChunkChoice> choices;
    time_t created;
    std::string model;
    std::string system_fingerprint;
    Usage usage;
    std::string id;

    // Additional fields
    Metric metric;
};

struct PrefixCache {
    std::vector<std::vector<uint16_t>> k_cache;
    std::vector<std::vector<uint16_t>> v_cache;
    std::vector<int> token_ids;
};

struct CnnChatData {
    std::string task_name;
    std::vector<uint8_t> data;
    std::vector<int64_t> data_shape;
    std::string data_type;
};

struct CnnChatCompletions: public ReqEntity {
    std::string case_name;
    int loop_num = 1;
    std::vector<CnnChatData> data_info;
};

using stream_cb_t = std::function<void(const ChatCompletionChunkObject &chunk)>;
