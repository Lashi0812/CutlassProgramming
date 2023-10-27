cmake_minimum_required(VERSION 3.26)
project(cutlass CUDA CXX)

find_package(CUDAToolkit REQUIRED)
include_directories("./third_party/cutlass/include")

file(GLOB CUDA_SOURCE_FILES "./csrc/*.cu")
foreach(CUDA_SOURCE_FILE ${CUDA_SOURCE_FILES})
    # Extract the filename without the extension
    get_filename_component(EXECUTABLE_NAME ${CUDA_SOURCE_FILE} NAME_WE)
    
    # Create an executable for each source file
    add_executable(${EXECUTABLE_NAME} ${CUDA_SOURCE_FILE})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89)
endforeach()

# add_executable(test "./csrc/intTuple.cu")
# set_target_properties(test PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89)