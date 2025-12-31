#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

enum class UserRequestType {
    INIT_MODEL,
    TERMINATE_MODEL,
    CHAT_COMPLETIONS,
    CHAT_COMPLETIONS_STREAM,
    ABORT_CHAT,
    LOAD_KV_CACHE,
    SAVE_KV_CACHE,

    CNN_CHAT_COMPLETIONS,
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
    std::vector<uint8_t> data;
    std::vector<int64_t> data_shape;
    std::string data_type;
};

struct CnnChatCompletions: public ReqEntity {
    std::string case_name;
    std::vector<CnnChatData> data_info;
};

using stream_cb_t = std::function<void(const ChatCompletionChunkObject &chunk)>;

/**
 * @brief initialize LLM Model
 * 
 * @details Initalize a LLM model specified by model, 
 *          model context length is specified by ctx_len. 
I*          By default, MiniCPM is initialized with context length is 4096
 * 
 * @param model LLM model, valid value: 
 *                  1:minicpm, (default)
 *                  2:deepseek
 *                  3:phi4
 *                  4:minicpmv1b
 *                  5:qwen
 *                  6:paligemma2
 *                  
 * @param model_root_path model root path
 * @param ctx_len max token context length 
 * @param system_prompt system prompt
 * @return return status specified by ErrCode
 * 
 */
// int init_model(int model = MINICPM, const std::string& model_root_path="", int ctx_len=4096);

/**
 * @brief run model with user specified prompt
 * 
 * @param input input data for the conversation
 * @param errcode_ret return an appropriate error code. if errcode_ret is null, no error code is returned.
 * @return response for user prompt.
 */
// ChatCompletionObject chat_completions(const ChatCompletionsRequest &request, int *errcode_ret);

/**
 * @brief run model with user specified prompt
 * 
 * @param input input data for the conversation
 * @param stream_cb callback function to handle response
 * @param errcode_ret return an appropriate error code. if errcode_ret is null, no error code is returned.
 */
// void chat_completions_stream(const ChatCompletionsRequest &request, stream_cb_t stream_cb, int *errcode_ret);

/**
 * @brief terminate LLM model service
 * 
 * @return ErrCode 
 */
int terminate_model();

/**
 * @brief Abort current conversation. This function does not block.
 * 
 */
void abort_request();

void enable_tracer(const std::string &filename);

void disable_tracer();

void use_rm_core_config_reg(int value);

// void set_min_prefill_npu_attn_token(uint32_t min_prefill_npu_attn_token);

void start_engine_server(const std::string &model_root_path);

void load_kv_cache(const PrefixCache &prefix_cache);

PrefixCache save_kv_cache(int len);
