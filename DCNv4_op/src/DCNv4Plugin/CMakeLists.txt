cmake_minimum_required(VERSION 3.15)
project(DCNv4Plugin LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS $ENV{TENSORRT_DIR}/include)
find_library(TENSORRT_LIBRARY nvinfer PATHS $ENV{TENSORRT_DIR}/lib)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin PATHS $ENV{TENSORRT_DIR}/lib)

include_directories(${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
add_library(dcnv4_plugin SHARED
    dcnv4_plugin.cpp
    plugin.cpp
    dcnv4_im2col_cuda.cuh
    dcnv4_col2im_cuda.cuh
    dcnv4_cuda.cu
)
target_link_libraries(dcnv4_plugin PRIVATE ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} ${CUDA_LIBRARIES})
set_target_properties(dcnv4_plugin PROPERTIES CUDA_ARCHITECTURES "80;86;89")
