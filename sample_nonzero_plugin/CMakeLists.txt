cmake_minimum_required(VERSION 3.18)
project(sample_non_zero_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ======= TensorRT 路径配置 =======
if(NOT DEFINED TENSORRT_DIR)
  set(TENSORRT_DIR "/usr/lib/TensorRT-10.13.3.9")
endif()

set(TRT_INCLUDE_DIR "${TENSORRT_DIR}/include")
if(EXISTS "${TENSORRT_DIR}/lib/x86_64-linux-gnu")
  set(TRT_LIB_DIR "${TENSORRT_DIR}/lib/x86_64-linux-gnu")
else()
  set(TRT_LIB_DIR "${TENSORRT_DIR}/lib")
endif()

message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib    : ${TRT_LIB_DIR}")

# ======= 依赖 =======
find_package(CUDAToolkit REQUIRED)

# 公共 include
set(COMMON_INCS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# 公共 CUDA 架构
set(COMMON_CUDA_ARCHS "60;70;75;80;86;89;90")

add_library(nonzero_plugin SHARED
  sample_nonzero_plugin.cc
  nonzero_kernel.cu
  common.cc
  plugin_register.cc
)

target_include_directories(nonzero_plugin PRIVATE ${COMMON_INCS})
set_target_properties(nonzero_plugin PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "${COMMON_CUDA_ARCHS}"
  OUTPUT_NAME "nonzero_plugin"   # 生成 libnonzero_plugin.so
)
target_link_directories(nonzero_plugin PRIVATE ${TRT_LIB_DIR})
target_link_libraries(nonzero_plugin
  PRIVATE
  nvinfer
  CUDA::cudart
)

# Linux 下方便运行
if(UNIX AND NOT APPLE)
  set_property(TARGET nonzero_plugin APPEND PROPERTY BUILD_RPATH ${TRT_LIB_DIR})
endif()

# ======= 可执行文件（你的示例程序）=======
set(SRCS_EXE
  common.cc
  build_engine.cc
  sample_nonzero_plugin.cc
  nonzero_kernel.cu
  main.cc
  plugin_register.cc
)
add_executable(${PROJECT_NAME} ${SRCS_EXE})
target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_INCS})
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "${COMMON_CUDA_ARCHS}"
)
target_link_directories(${PROJECT_NAME} PRIVATE ${TRT_LIB_DIR})
target_link_libraries(${PROJECT_NAME}
  PRIVATE
  nvinfer
  CUDA::cudart
)

if(UNIX AND NOT APPLE)
  set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY BUILD_RPATH ${TRT_LIB_DIR})
endif()
