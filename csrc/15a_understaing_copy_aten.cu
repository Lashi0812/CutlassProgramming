#include "cute/stride.hpp"
#include "cute/tensor.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cute/pointer.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/numeric/half.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/algorithm/copy.hpp"
#include <ATen/ATen.h>
#include <iostream>
#include <vector>
#include "cute/util/debug.hpp"
#include "latex.hpp"

using namespace cute;

template <typename TiledCopy, typename T>
__global__ void copy_kernel(T const *in, T *out) {

    TiledCopy tiled_copy;
    __shared__ T smemA[cosize_v<Layout<typename TiledCopy::TiledShape_MN>>];

    auto gA = make_tensor(make_gmem_ptr(in), typename TiledCopy::TiledShape_MN{});
    auto sA = make_tensor(make_smem_ptr(smemA), typename TiledCopy::TiledShape_MN{});
    auto gC = make_tensor(make_gmem_ptr(out), typename TiledCopy::TiledShape_MN{});

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
    auto tAgA = thr_copy.partition_S(gA);
    auto tAsA = thr_copy.partition_D(sA);
    auto tAgC = thr_copy.partition_S(gC);

    auto tArA = make_fragment_like(tAsA);

    // if (thread0()) {
    //     // clang-format off
    //     print(gA);print("\n");
    //     print(sA);print("\n");
    //     print(tAgA);print("\n");
    //     print(tAsA);print("\n");
    //     print("Common Vector : ");print(max_common_vector(tAgA, tAsA));print("\n");
    //     // clang-format on
    // }
    copy(tAgA, tAsA);
    cp_async_fence();
    cp_async_wait<0>();
    
    copy(tAsA, tArA);

    transform(tArA, pre_increment{});

    copy(tArA, tAgC);
}

void test_copy_host() {

    // using TiledCopy = decltype(make_tiled_copy(
    //   Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    //   Layout<Shape<_2, _16>>{},
    //   Layout<Shape<_8, _1>>{}));

    using TiledCopy = decltype(
    make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_8>>{}));

    TiledCopy tiled_copy;
    std::vector<int> vecG(1024);
    std::vector<int> vecS(1024);
    // auto gA = make_counting_tensor(Layout<typename TiledCopy::TiledShape_MN>{});
    // auto sA = make_counting_tensor(Layout<typename TiledCopy::TiledShape_MN>{});
    auto gA = make_tensor(vecG.data(), Layout<typename TiledCopy::TiledShape_MN,Stride<_64,_1>>{});
    auto sA = make_tensor(vecS.data(), Layout<typename TiledCopy::TiledShape_MN,Stride<_64,_1>>{});

    auto thr_copy = tiled_copy.get_slice(0);
    auto tAgA = thr_copy.partition_S(gA);
    auto tAsA = thr_copy.partition_D(sA);

    // clang-format off
    print(gA);print("\n");
    print(sA);print("\n");
    print(tAgA);print("\n");
    print(tAsA);print("\n");
    print("Common Vector : ");print(max_common_vector(tAgA, tAsA));print("\n");
    // clang-format on

    copy(tAgA, tAsA);
}

void test_normal_copy() {
    using tiled_copy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_2, _16>>{},
      Layout<Shape<_8, _1>>{}));

    auto in =
      at::arange(
        decltype(size(tiled_copy::TiledShape_MN{}))::value, at::TensorOptions().dtype(at::kHalf))
        .reshape({size<0>(tiled_copy::TiledShape_MN{}), size<1>(tiled_copy::TiledShape_MN{})});

    auto out = at::zeros_like(in);

    half_t *d_in, *d_out;
    cudaMalloc((void **)&d_in, in.numel() * in.element_size());
    cudaMalloc((void **)&d_out, out.numel() * out.element_size());

    cudaMemcpy(d_in, in.data_ptr(), in.numel() * in.element_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out.data_ptr(), out.numel() * out.element_size(), cudaMemcpyHostToDevice);

    copy_kernel<tiled_copy><<<1, 32>>>(d_in, d_out);
    cudaMemcpy(out.data_ptr(), d_out, out.numel() * out.element_size(), cudaMemcpyDeviceToHost);
    std::cout << out << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    // print_latex(tiled_copy{}, "CP_T16x2_V1x8", 3);
}

int main() {
    test_normal_copy();
    cudaDeviceReset();
    // test_copy_host();
}