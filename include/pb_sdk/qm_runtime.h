#pragma once

#include <atomic>
#include <variant>
#include <vector>
#include <stdint.h>

#include "pb_infer_api.h"

/**
 * @brief 每个NN调用函数所返回的错误码
 */
typedef enum {
    PBNN_SUCCESS = 0,                       // 正确情况
    PBNN_INVALID_ARGUMENT = -6000001,
    PBNN_INVALID_MODEL = -6000002,
    PBNN_INVALID_MODEL_HANDLE = -6000003,
    PBNN_INVALID_FILE = -6000004,
    PBNN_OUT_OF_MEMORY = -6000005,
    PBNN_TIMEOUT = -6000006,
    PBNN_DISCONNECT = -6000007,
    PBNN_INIT_FAILED = -6000008,
} PBNNERRCODE;

class ModelHandler
{
public:
    ModelHandler();
    ~ModelHandler();

    /**
     * @brief 模型初始化
     * 
     * @param [in]model 模型类型
     * @param [in]model_path 模型pbnn文件路径
     * @param [in]ctx_len 最大上下文长度
     * 
     * @return 错误码
     */
    int init(int model, const std::string& model_path, int ctx_len = 4096);
    /**
     * @brief llm输入对话请求
     * 
     * @param [in]request 对话请求信息
     * @param [in]is_stram 是否为流式输入
     */
    void input(const ChatCompletionsRequest &request, bool is_stream);
    /**
     * @brief cnn输入对话请求
     * 
     * @param [in]request 对话请求信息
     */
    void input(const CnnChatCompletions &request);
    /**
     * @brief 执行对话
     * 
     * @return 错误码
     */
    int execute();
    /**
     * @brief 获取对话输出
     * 
     * @return 对话响应信息
     */
    std::variant<ChatCompletionObject, ChatCompletionChunkObject, CnnChatCompletions> output();

    /**
     * @brief 获取连接状态
     * 
     * @return true-已连接, false-未连接
     */
    bool is_connected();

private:
    /**
     * @brief 连接框架服务器
     */
    void connect_infer_server();
    /**
     * @brief 接收数据
     */
    std::vector<uint8_t> recv_data();
    /**
     * @brief 发送数据
     */
    int send_data(const std::vector<uint8_t>& data);

private:
    int m_client_fd;
    int model_type;

    std::atomic<bool> m_have_output;
    std::atomic<bool> m_execute_llm;
    std::atomic<bool> m_connected;

    bool                      m_stream;
    std::vector<uint8_t>      m_request;
    ChatCompletionObject      m_response;
    ChatCompletionChunkObject m_response_stream;
    CnnChatCompletions        m_cnn_response;
};

