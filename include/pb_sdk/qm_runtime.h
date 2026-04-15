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
    PBNN_SUCCESS              = 0,        // 正确情况
    PBNN_INVALID_ARGUMENT     = -6000001, // 入参错误
    PBNN_INVALID_MODEL        = -6000002, // 无效模型
    PBNN_INVALID_MODEL_HANDLE = -6000003, // 无效模型句柄
    PBNN_INVALID_FILE         = -6000004, // 无效文件
    PBNN_OUT_OF_MEMORY        = -6000005, // 内存溢出
    PBNN_TIMEOUT              = -6000006, // 超时
    PBNN_DISCONNECT           = -6000007, // 断开连接
    PBNN_INIT_FAILED          = -6000008, // 初始化失败
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
    int init(const std::string& model_name, const std::string& model_path, int ctx_len = 4096);
    int unload();
    /**
     * @brief 获取连接状态
     * 
     * @return true-已连接, false-未连接
     */
    bool is_connected();

    /**
     * @brief 输入对话请求
     * 
     * @param [in]request 对话请求信息
     */
    void input(const std::variant<ChatCompletionsRequest, CnnChatCompletions> &request);
    /**
     * @brief 执行对话
     * 
     * @param [in]stream_cb 入参非空时，表示执行流式对话(仅在llm模型中有效)
     * @return 错误码
     */
    int execute(stream_cb_t stream_cb);
    /**
     * @brief 获取llm(非流式)和cnn对话输出
     * 
     * @return 对话响应信息
     */
    std::variant<ChatCompletionObject, CnnChatCompletions> output();

    int load_kv_cache(const PrefixCache &prefix_cache);

    PrefixCache save_kv_cache(int len);
    
    void start_trace(const std::string& trace_file);
    void stop_trace();

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
    std::string m_model_name;

    std::atomic<bool> m_have_output;
    std::atomic<bool> m_execute_llm;
    std::atomic<bool> m_connected;

    std::vector<uint8_t>    m_request;
    ChatCompletionObject    m_response;
    CnnChatCompletions      m_cnn_response;
};

