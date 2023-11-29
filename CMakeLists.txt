cmake_minimum_required(VERSION 3.26)
project(cutlass CUDA CXX)

set(CMAKE_PREFIX_PATH /root/miniconda3/envs/nlp/lib/python3.11/site-packages/torch/share/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)


find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(
    "third-party/cutlass/include"
    "include"
)

file(GLOB CUDA_SOURCE_FILES "./csrc/*.cu")
foreach(CUDA_SOURCE_FILE ${CUDA_SOURCE_FILES})
    # Extract the filename without the extension
    get_filename_component(EXECUTABLE_NAME ${CUDA_SOURCE_FILE} NAME_WE)
    string(FIND ${EXECUTABLE_NAME} "aten" containsAten)    

    # Create an executable for each source file
    add_executable(${EXECUTABLE_NAME} ${CUDA_SOURCE_FILE})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89 CUDA_SEPARABLE_COMPILATION ON)
    if(containsAten GREATER -1)
        message(STATUS "Linking torch libraries to ${EXECUTABLE_NAME}")
        target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")
    endif()
    # target_compile_options(${EXECUTABLE_NAME} PRIVATE -G -g)
endforeach()

# add_executable(test "./csrc/intTuple.cu")
# set_target_properties(test PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89)