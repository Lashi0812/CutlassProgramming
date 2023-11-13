#include <ATen/ATen.h>
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

__global__ void test_block_manual_access(const int *__restrict__ a)
{
    auto tensor = make_tensor(make_gmem_ptr(a), make_shape(make_shape(blockDim.x, gridDim.x), make_shape(blockDim.y, gridDim.y)));

    for (int blk{0}; blk < gridDim.x * gridDim.y; ++blk)
        for (int thr{0}; thr < blockDim.x * blockDim.y; ++thr)
        {
            if (thread(thr, blk))
            {
                print("Block id %d , thread %d and Element at (%d,%d),(%d,%d) : %d\n",
                      blk, thr,
                      blockIdx.x, blockIdx.y,
                      threadIdx.x, threadIdx.y,
                      tensor(make_coord(make_coord(threadIdx.x, blockIdx.x), make_coord(threadIdx.y, blockIdx.y))));
            }
        }
}

__global__ void test_zipped_access(const int *__restrict__ a)
{
    auto tensor = make_tensor(make_gmem_ptr(a), make_shape(make_shape(blockDim.x, gridDim.x), make_shape(blockDim.y, gridDim.y)));
    auto zipped = zipped_divide(tensor, make_shape(blockDim.x, blockDim.y));

    for (int blk{0}; blk < size<1>(zipped); ++blk)
        for (int thr{0}; thr < size<0>(zipped); ++thr)
        {
            if (thread(thr, blk))
            {
                print("Block id %d , thread %d : %d\n",
                      blk, thr, zipped(make_coord(make_coord(threadIdx.x, threadIdx.y), make_coord(blockIdx.x, blockIdx.y))));
            }
        }
}

int main()
{
    auto a = at::arange(24, at::TensorOptions().dtype(at::kInt));

    int *d_a;
    cudaMalloc((void **)&d_a, a.numel() * a.element_size());
    cudaMemcpy(d_a, a.data_ptr(), a.numel() * a.element_size(), cudaMemcpyHostToDevice);

    dim3 block(2, 3);
    dim3 grid(2, 2);

    // test_block_manual_access<<<grid, block>>>(d_a);
    test_zipped_access<<<grid, block>>>(d_a);

    cudaFree(d_a);
    cudaDeviceReset();
}