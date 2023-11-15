/*
Copy data from global to shared and back to global
*/

#include <ATen/ATen.h>
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

template <typename T, typename MShape, typename NShape, typename BLKLayout>
__global__ static void copy_kernel(T const *in, T *out,
                                   MShape M, NShape N,
                                   BLKLayout tiler)
{
    __shared__ T smem[cosize_v<BLKLayout>];
    T reg{0};

    auto la = make_layout(make_shape(M, N));
    auto tensorA = make_tensor(make_gmem_ptr(in), la);
    auto tensorC = make_tensor(make_gmem_ptr(out), la);

    auto sA = make_tensor(make_smem_ptr(smem), tiler);
    auto rA = make_tensor(make_rmem_ptr(&reg),make_shape(_1{},_1{}));

    auto blockA = local_tile(tensorA, shape(tiler), make_coord(blockIdx.x, blockIdx.y));
    auto tAgA = local_partition(blockA, tiler, threadIdx.x);
    auto tAsA = local_partition(sA, tiler, threadIdx.x);

    auto blockC = local_tile(tensorC, shape(tiler), make_coord(blockIdx.x, blockIdx.y));
    auto tCgC = local_partition(blockC, tiler, threadIdx.x);

    copy(tAgA, tAsA);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    copy(tAsA, rA);
    reg += 10;
    copy(rA,tCgC);

    // if(thread(2))
    // {
    //     print_tensor(tensorA);
    //     print_tensor(blockA);
    //     print_tensor(sA);
    //     print_tensor(tAgA);
    //     print_tensor(tAsA);
    // }
}

void test_global_shared_global(
    at::Tensor &in, at::Tensor &out,
    int *d_in, int *d_out,
    int m, int n)
{
    auto M = int(m);
    auto N = int(n);

    auto BM = Int<16>{};
    auto BN = Int<16>{};

    auto tile_layout = make_layout(make_shape(BM, BN));

    dim3 block(size(tile_layout));
    dim3 grid(ceil_div(size(M), BM), ceil_div(size(N), BN));
    cudaMemset(d_out, 0, out.numel() * out.element_size());

    std::cout << block << std::endl;
    std::cout << grid << std::endl;
    copy_kernel<<<grid, block>>>(d_in, d_out, M, N, tile_layout);

    cudaMemcpy(out.data_ptr(), d_out, out.numel() * out.element_size(), cudaMemcpyDeviceToHost);
    // std::cout << in << std::endl;
    // std::cout << out << std::endl;
    std::cout << ((out.equal(in.add(10))) ? "Copy Success" : "Copy Failed") << std::endl;
}

int main()
{
    int M = Int<4096>{};
    int N = Int<4096>{};

    auto in = at::arange(M * N, at::TensorOptions().dtype(at::kInt)).reshape({M, N});
    auto out = at::zeros_like(in);

    // allocate
    auto nBytes = in.numel() * in.element_size();
    int *d_in, *d_out;

    cudaMalloc((void **)&d_in, nBytes);
    cudaMalloc((void **)&d_out, nBytes);

    cudaMemcpy(d_in, in.data_ptr(), nBytes, cudaMemcpyHostToDevice);

    test_global_shared_global(in, out, d_in, d_out, M, N);

    cudaFree(d_in);
    cudaFree(d_out);

    cudaDeviceReset();
}