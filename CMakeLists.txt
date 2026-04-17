cmake_minimum_required(VERSION 3.16)
project(model-zoo C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# 交叉编译请使用:
# cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-aarch64.cmake

# =============================
# ONNX Runtime (aarch64 预编译)
# =============================
set(ORT_ROOT "/host/cc_for_aarch64/third_parties/onnxruntime-linux-aarch64")
set(ORT_INCLUDE "${ORT_ROOT}/include")
set(ORT_LIB     "${ORT_ROOT}/lib")


set(Torch_ROOT /home/fuwj691/libtorch-arm64-abi1-cpu-silent/)
set(Torch_DIR  "${Torch_ROOT}/share/cmake/Torch")

# =============================
# PyTorch (libtorch)
# =============================
find_package(Torch REQUIRED)

# =============================
# 收集预编译库
# =============================
file(GLOB LIB_KWS       CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/kws/*.so*")
file(GLOB LIB_KWS_A    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/kws/*.a")
file(GLOB LIB_TEN_VAD  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/ten_vad/*.so*")
file(GLOB LIB_PBSDK    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/pb_sdk/*.so*")

set(ALL_LIBS
    ${LIB_KWS}
    ${LIB_KWS_A}
    ${LIB_TEN_VAD}
    ${LIB_PBSDK}
)

# =============================
# 源文件
# =============================
set(SOURCES
    src/main.cpp
    src/kws/kws.cpp
    src/vad/vad.cpp
)

# =============================
# 构建可执行文件
# =============================
add_executable(model-zoo ${SOURCES})

# =============================
# Include 目录
# =============================
target_include_directories(model-zoo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pb_sdk
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kws
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ten_vad
    ${ORT_INCLUDE}                                   # onnxruntime headers
    ${TORCH_INCLUDE_DIRS}                            # libtorch headers
)

# =============================
# 链接库
# =============================
target_link_libraries(model-zoo PRIVATE
    ${ALL_LIBS}
    ${TORCH_LIBRARIES}
    pthread
    dl
    m
)

# rpath-link 帮助链接器找到间接依赖的 .so
target_link_options(model-zoo PRIVATE
    "LINKER:-rpath-link,${CMAKE_CURRENT_SOURCE_DIR}/lib/pb_sdk"
    "LINKER:-rpath-link,${CMAKE_CURRENT_SOURCE_DIR}/lib/kws"
    "LINKER:-rpath-link,${CMAKE_CURRENT_SOURCE_DIR}/lib/ten_vad"
    "LINKER:-rpath-link,${ORT_LIB}"
)

# =============================
# RPATH — 运行时在板上查找 .so
# =============================
set_target_properties(model-zoo PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib/kws:$ORIGIN/lib/pb_sdk:$ORIGIN/lib/ten_vad"
    BUILD_WITH_INSTALL_RPATH ON
)
