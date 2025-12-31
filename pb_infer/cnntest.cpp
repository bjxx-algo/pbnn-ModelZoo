#include "nlohmann/json.hpp"
#include "pb_infer_api.h"
#include "qm_runtime.h"
#include <algorithm>
#include <assert.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <exception>
#include <getopt.h>
#include <iostream>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <sys/types.h>
#include <unordered_map>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using namespace std;

static struct {
    std::string model_root_path{"/data/models/pbnn"};
    std::string test_results_path{"./cnn_results.json"};
} options;

static std::unordered_map<int, std::string> model_files= {
    {RESNET50, "int8_resnet50_sim_b1.pbnn"},
    {REPVGG, "int8_repvgg_b1_sim_b1.pbnn"},
    {YOLOV8S, "yolov8s.pbnn"},
};

struct Similarity {
    double mse;
    double consine_sim;
    double max_relative;
    double max_abs;
};

static int test_case_idx = 0;
static nlohmann::ordered_json test_results;

void run_test_file(const std::string &config_filename);
void run_test_case(const nlohmann::json test_case, const fs::path &config_dir);

// 软件实现的fp16到double转换
double fp16_to_fp64_soft(uint16_t fp16) {
    // 提取符号、指数、尾数
    uint32_t sign = (fp16 >> 15) & 0x1;
    uint32_t exponent = (fp16 >> 10) & 0x1F;
    uint32_t mantissa = fp16 & 0x3FF;
    
    uint64_t double_bits;
    
    if (exponent == 0) {
        // 零或非规约数
        if (mantissa == 0) {
            // 零
            double_bits = (uint64_t)sign << 63;
        } else {
            // 非规约数
            float value = (float)mantissa / 1024.0f * powf(2.0f, -14.0f);
            if (sign) value = -value;
            return (double)value;
        }
    } else if (exponent == 0x1F) {
        // 无穷大或NaN
        double_bits = ((uint64_t)sign << 63) | (0x7FFULL << 52) | ((uint64_t)mantissa << 42);
    } else {
        // 规约数
        uint64_t double_exponent = (uint64_t)(exponent - 15 + 1023);
        uint64_t double_mantissa = (uint64_t)mantissa << 42;
        double_bits = ((uint64_t)sign << 63) | (double_exponent << 52) | double_mantissa;
    }
    
    return *(double*)&double_bits;
}


template<typename T>
std::vector<T> convert_nchw_to_nhwc(const std::vector<T>& input_data, 
                                          int N, int C, int H, int W) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        throw std::invalid_argument("Dimensions must be positive integers");
    }
    
    size_t total_elements = static_cast<size_t>(N) * C * H * W;
    if (input_data.size() != total_elements) {
        throw std::invalid_argument("Input data size does not match the specified dimensions");
    }
    
    std::vector<T> output_data(total_elements);
    
    // 优化版：调整循环顺序以获得更好的缓存性能[6](@ref)
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    size_t nchw_index = n * C * H * W + c * H * W + h * W + w;
                    size_t nhwc_index = n * H * W * C + h * W * C + w * C + c;
                    output_data[nhwc_index] = input_data[nchw_index];
                }
            }
        }
    }
    
    return output_data;
}

template<typename T>
std::vector<T> convert_nhwc_to_nchw(const std::vector<T>& input_data,
                               int N, int C, int H, int W) {
    // 参数校验
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        throw std::invalid_argument("Dimensions must be positive integers");
    }
    
    size_t total_elements = static_cast<size_t>(N) * C * H * W;
    if (input_data.size() != total_elements) {
        throw std::invalid_argument("Input data size does not match the specified dimensions");
    }
    
    std::vector<T> output_data(total_elements);
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    size_t nhwc_index = n * H * W * C + h * W * C + w * C + c;
                    size_t nchw_index = n * C * H * W + c * H * W + h * W + w;
                    output_data[nchw_index] = input_data[nhwc_index];
                }
            }
        }
    }
    
    return output_data;
}

template<typename T>
std::vector<T> convert_byte_to_type(const std::vector<uint8_t>& bytes) {
    if (bytes.empty()) {
        throw std::runtime_error("字节流为空");
    }
    
    // 检查字节数是否是类型大小的整数倍
    if (bytes.size() % sizeof(T) != 0) {
        std::stringstream ss;
        ss << "字节数(" << bytes.size() << ") 不是类型大小(" << sizeof(T) << ") 的整数倍";
        throw std::runtime_error(ss.str());
    }
    
    size_t element_count = bytes.size() / sizeof(T);
    std::vector<T> result(element_count);
    
    // 复制字节到结果向量
    memcpy(result.data(), bytes.data(), bytes.size());
    
    return result;
}

double calc_mse(const std::vector<double>& data1, const std::vector<double>& data2) {
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("data size mismatch");
    }
    
    double sumSquaredErrors = 0.0;
    for (size_t i = 0; i < data1.size(); ++i) {
        double error = data1[i] - data2[i];
        sumSquaredErrors += error * error;
    }
    
    return sumSquaredErrors / data1.size();
}

double calc_cosine_sim(const std::vector<double>& data1, const std::vector<double>& data2) {
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("data size mismatch");
    }
    
    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    
    for (size_t i = 0; i < data1.size(); ++i) {
        dotProduct += data1[i] * data2[i];
        norm1 += data1[i] * data1[i];
        norm2 += data2[i] * data2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 < std::numeric_limits<double>::epsilon() || 
        norm2 < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("向量模长为零，无法计算余弦相似度");
    }
    
    return dotProduct / (norm1 * norm2 + std::numeric_limits<double>::epsilon());
}

double calc_max_abs_error(const std::vector<double>& data1, const std::vector<double>& data2) {
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("data size mismatch");
    }
    
    double maxError = 0.0;
    for (size_t i = 0; i < data1.size(); ++i) {
        double error = std::abs(data1[i] - data2[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    
    return maxError;
}

double calc_max_relative_error(const std::vector<double>& data1, const std::vector<double>& data2) {
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("data size mismatch");
    }
    
    double maxRelativeError = 0.0;
    int validCount = 0;
    
    for (size_t i = 0; i < data1.size(); ++i) {
        // 避免除以零，只有当data1[i]不为零时才计算相对误差
        if (std::abs(data1[i]) > std::numeric_limits<double>::epsilon()) {
            double relativeError = std::abs(data1[i] - data2[i]) / std::abs(data1[i]);
            if (relativeError > maxRelativeError) {
                maxRelativeError = relativeError;
            }
            validCount++;
        }
    }
    
    if (validCount == 0) {
        throw std::runtime_error("所有参考值都为零，无法计算相对误差");
    }
    
    return maxRelativeError;
}

template<typename T>
bool load_binary_file(const std::string& filename, std::vector<T>& data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 读取数据
    data.resize(fileSize / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();

    // std::cout << "Read " << data.size() << " elements from " << filename << std::endl;
    if (!data.empty()) {
        std::cout << "First element: " << data[0] << std::endl;
    }

    return true;
}

void load_input(const string& file, const string data_type, std::vector<int64_t>& shape, std::vector<uint8_t>& output) {
    assert(shape.size() == 4);

    if (data_type == "float16") {
        std::vector<uint16_t> nchw_data;
        load_binary_file(file, nchw_data);
        auto nhwc_data = convert_nchw_to_nhwc(nchw_data, shape[0], shape[1], shape[2], shape[3]);
        output.resize(nhwc_data.size() * sizeof(uint16_t));
        memcpy(output.data(), reinterpret_cast<uint8_t*>(nhwc_data.data()), output.size());
    } else if (data_type == "uint8_t") {
        std::vector<uint8_t> nchw_data;
        load_binary_file(file, nchw_data);
        output = convert_nchw_to_nhwc(nchw_data, shape[0], shape[1], shape[2], shape[3]);
    } else {
        std::cerr << "unsuported binary data type: " << data_type << std::endl;
    }
}

Similarity verify_fp16_data(std::vector<uint16_t>& data1, std::vector<uint16_t>& data2) {
    std::vector<double> data1_cvt(data1.size());
    std::transform(data1.begin(), data1.end(), data1_cvt.begin(), fp16_to_fp64_soft);
    std::vector<double> data2_cvt(data2.size());
    std::transform(data2.begin(), data2.end(), data2_cvt.begin(), fp16_to_fp64_soft);
    Similarity result;
    result.mse = calc_mse(data1_cvt, data2_cvt);
    result.consine_sim = calc_cosine_sim(data1_cvt, data2_cvt);
    result.max_abs = calc_max_abs_error(data1_cvt, data2_cvt);
    result.max_relative = calc_max_relative_error(data1_cvt, data2_cvt);

    return result;
}

Similarity verify_data(const CnnChatData& data, const std::string golden_file, const std::vector<int64_t>& golden_shape) {
    assert(data.data_shape == golden_shape);
    if (data.data_type == "float16") {
        std::vector<uint16_t> golden_data;
        load_binary_file(golden_file, golden_data);
        std::vector<uint16_t> nhwc(data.data.size()/ sizeof(uint16_t));
        memcpy(reinterpret_cast<uint8_t*>(nhwc.data()), data.data.data(), data.data.size());
        auto nchw = convert_nhwc_to_nchw(nhwc, data.data_shape[0], data.data_shape[1], data.data_shape[2], data.data_shape[3]);
        return verify_fp16_data(nchw, golden_data);
    } else if (data.data_type == "uint8_t") {
        //TODO:
    }

    return Similarity();
}

int main(int argc, char* argv[]) {
    while (true) {
        static struct option long_options[] = {
            {"help", no_argument, 0, 'h'},
            {"model-root-path", required_argument, 0, 0},
            {0, 0, 0, 0}
        };
        int option_index = 0;
        int c = getopt_long(argc, argv, "ho:", long_options, &option_index);
        if (c == -1) {
            break;
        }

        switch (c) {
        case 'h':
            std::cout << "Usage: " << argv[0] << " [options] config_file..." << std::endl;
            std::cout << R"(Options:
  -h, --help                      Display this information
      --model-root-path=PATH      Model root path
  -o FILE                         Output test results to FILE (default: ./test_results.json)
)";
            return 0;
        case 'o':
            options.test_results_path = optarg;
            break;
        case 0:
            if (long_options[option_index].name == std::string("model-root-path")) {
                options.model_root_path = optarg;
            } else {
                std::cerr << "Unknown option: " << long_options[option_index].name << std::endl;
                return 1;
            }
            break;
        default:
            std::cerr << "Try '" << argv[0] << " --help' for more information." << std::endl;
            return 1;
        }
    }

    if (optind == argc) {
        std::cerr << "Usage: " << argv[0] << " [options] config_file..." << std::endl;
        return 1;
    }
    for (int i = optind; i < argc; i++) {
        run_test_file(argv[i]);
    }

    std::ofstream results_file(options.test_results_path);
    if (!results_file) {
        std::cerr << "Failed to open test results file " << options.test_results_path << ": " << strerror(errno) << std::endl;
        return 1;
    }
    results_file << test_results.dump(2) << std::endl;
    std::cout << "Test results written to " << options.test_results_path << std::endl;
    return 0;
}

void run_test_file(const std::string &config_filename) {
    std::ifstream config_file(config_filename);
    if (!config_file) {
        std::cerr << "Failed to open config file " << config_filename << ": " << strerror(errno) << std::endl;
        return;
    }
    auto config_dir = fs::path(config_filename).parent_path();
    try {
        nlohmann::json config;
        config_file >> config;

        for (const auto &include_file : config.value("include", nlohmann::json::array())) {
            run_test_file(config_dir / include_file);
        }
        for (const auto &test_case : config.value("cases", nlohmann::json::array())) {
            run_test_case(test_case, config_dir);
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

void run_test_case(const nlohmann::json test_case, const fs::path& config_dir) {
    try {
        int model_id = test_case.at("model_id");
        std::string name = test_case.value("name", "Unamed");
        std::string model_path = options.model_root_path + "/" + model_files[model_id];
        ModelHandler model;
        model.init(model_id, model_path);
        CnnChatCompletions request;
        request.case_name = name;
        for (const auto& input: test_case.at("inputs")) {
            const std::string& input_type = input.at("type");
            CnnChatData part;
            if (input_type == "pixel") {
                part.data_type = input.at("data_type");
                input.at("shape").get_to(part.data_shape);
                std::string input_path = config_dir/input.at("pixel_file");
                load_input(input_path, part.data_type, part.data_shape, part.data); 
                request.data_info.push_back(std::move(part));
            } else if (input_type == "image") {
                //TODO:
            } else {
                std::cerr << "unknown cnn input type: " << input_type << std::endl;
                throw std::runtime_error("Unknown CNN input type:" + input_type);
            }
        }
        model.input(request);
        model.execute();
        auto ret = model.output();
        auto result = std::get<CnnChatCompletions>(ret);
        ssize_t output_id = 0;
        nlohmann::json details;
        for (const auto& golden: test_case.at("golden")) {
            std::string golden_file = config_dir/golden.at("file");
            std::vector<int64_t> golden_shape;
            golden.at("shape").get_to(golden_shape);
            auto cmp = verify_data(result.data_info[output_id], golden_file, golden_shape);
            details.push_back({
                {"output index", output_id},
                {"mse", cmp.mse},
                {"consine_sim", cmp.consine_sim},
                {"max abs error", cmp.max_abs},
                {"max relative error", cmp.max_relative}
            });
        }
        
        test_results["cases"].push_back({
            {"index", test_case_idx},
            {"name", name},
            {"result", details}
        });


    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::string name = test_case.value("name", "Unnamed");
        test_results["cases"].push_back({
            {"index", test_case_idx},
            {"name", name},
            {"error", e.what()}
        });
    }
    test_case_idx++;
}