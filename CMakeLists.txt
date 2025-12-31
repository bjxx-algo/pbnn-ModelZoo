cmake_minimum_required(VERSION 3.16)
project(yolov8s_demo C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    add_compile_options(-mfpu=neon-fp16)
endif()
file(GLOB _all_so "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.so*")
list(FILTER _all_so EXCLUDE REGEX "\\.so\\.[0-9]+\\.[0-9]+$")

add_executable(yolov8_demo src/main.cpp)
target_include_directories(yolov8_demo PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(yolov8_demo PRIVATE ${_all_so})

