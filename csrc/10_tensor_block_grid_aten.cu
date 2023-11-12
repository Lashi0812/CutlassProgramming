#include <ATen/ATen.h>
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

__global__ void test(const int *__restrict__ a)
{
    auto tensor = make_tensor(make_gmem_ptr(a), make_shape(make_shape(blockDim.x, gridDim.x), make_shape(blockDim.y, gridDim.y)));
    if (thread0())
    {
        print_tensor(tensor);
    }
    print("Element at (%d,%d),(%d,%d) : %d\n",
          blockIdx.x, blockIdx.y,
          threadIdx.x, threadIdx.y,
          tensor(make_coord(make_coord(threadIdx.x, blockIdx.x), make_coord(threadIdx.y, blockIdx.y))));
}

int main()
{
    auto a = at::arange(16,at::TensorOptions().dtype(at::kInt));

    std::cout << a << std::endl;

    int *d_a;
    cudaMalloc((void **)&d_a, a.numel() * a.element_size());
    cudaMemcpy(d_a, a.data_ptr(), a.numel() * a.element_size(), cudaMemcpyHostToDevice);

    dim3 block(2, 2);
    dim3 grid(2, 2);

    test<<<grid, block>>>(d_a);
    cudaFree(d_a);
    cudaDeviceReset();
}