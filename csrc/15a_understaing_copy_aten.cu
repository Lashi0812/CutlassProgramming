#include "cute/swizzle.hpp"
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
#include "cute/util/debug.hpp"
#include <ATen/ATen.h>
#include <iostream>
#include <numeric>

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////
//                      Generic Copy using LD
///////////////////////////////////////////////////////////////////////////////////////////////

template <typename TiledCopy, typename T>
__global__ void copy_kernel(T const *in, T *out) {

    TiledCopy tiled_copy;
    __shared__ T smemA[cosize_v<Layout<typename TiledCopy::TiledShape_MN>>];

    auto swizzle_layout = composition(Swizzle<2, 3, 3>{}, Layout<Shape<_16, _16>>{});
    // auto smem_layout = tile_to_shape(swizzle_layout, Shape<_16, _16>{});
    auto gA = make_tensor(make_gmem_ptr(in), typename TiledCopy::TiledShape_MN{});
    // auto sA = make_tensor(make_smem_ptr(smemA), typename TiledCopy::TiledShape_MN{});

    auto sA = make_tensor(make_smem_ptr(smemA), swizzle_layout);
    auto gC = make_tensor(make_gmem_ptr(out), typename TiledCopy::TiledShape_MN{});

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
    auto tAgA = thr_copy.partition_S(gA);
    auto tAsA = thr_copy.partition_D(sA);
    auto tAgC = thr_copy.partition_S(gC);

    auto tArA = make_fragment_like(tAsA);

    copy(tiled_copy, tAgA, tAsA);
    cp_async_fence();
    cp_async_wait<0>();

    // print_helper(tAgA, tAsA);

    // clang-format off
        // print(gA);print("\n");
        // print(sA);print("\n");
        // print(tAgA);print("\n");
        // print(tAsA);print("\n");
        // print("Common Vector GS : ");print(max_common_vector(tAgA, tAsA));print("\n");
        // print("Common Vector SR : ");print(max_common_vector(tAsA, tArA));print("\n");
        // print("Common Vector RG : ");print(max_common_vector(tArA, tAgC));print("\n");
    // clang-format on

    // for (int i{0}; i < size(tAgA); ++i)
    //     tAsA(i) = tAgA(i);
    // copy(Copy_Atom<UniversalCopy<uint32_t>,half_t>{},tAgA, tAsA);

    copy(tAsA, tArA);
    // for(int i{0};i<size(tAsA);++i)
    //   tArA(i) = tAsA(i);

    transform(tArA, pre_increment{});

    copy(tArA, tAgC);
}

void test_copy_host() {

    // using TiledCopy = decltype(make_tiled_copy(
    //   Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    //   Layout<Shape<_2, _16>>{},
    //   Layout<Shape<_8, _1>>{}));

    using TiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint32_t>, half_t>{},
      Layout<Shape<_2, _16>>{},
      Layout<Shape<_8, _1>>{}));

    TiledCopy tiled_copy;
    std::vector<int> vecG(256);
    std::vector<int> vecS(256);
    std::iota(vecG.begin(), vecG.end(), 0);
    std::iota(vecS.begin(), vecS.end(), 0);

    auto swizzle_layout = composition(Swizzle<3, 1, 5>{}, Layout<Shape<_16, _16>>{});
    // auto smem_layout = tile_to_shape(swizzle_layout, Shape<_16, _16>{});
    // auto gA = make_counting_tensor(Layout<typename TiledCopy::TiledShape_MN>{});
    // auto sA = make_counting_tensor(Layout<typename TiledCopy::TiledShape_MN>{});
    auto gA = make_tensor(vecG.data(), Layout<typename TiledCopy::TiledShape_MN>{});
    auto sA = make_tensor(vecS.data(), swizzle_layout);

    auto thr_copy = tiled_copy.get_slice(8);
    auto tAgA = thr_copy.partition_S(gA);
    auto tAsA = thr_copy.partition_D(sA);

    // copy(tAgA, tAsA);

    // clang-format off
    print(gA);print("\n");
    print(sA);print("\n");print_tensor(sA);    
    print(tAgA);print("\n");print_tensor(tAgA);
    print(tAsA);print("\n");print_tensor(tAsA);
    print("Common Vector : ");print(max_common_vector(tAgA, tAsA));print("\n");
    // clang-format on
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
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                          ldmatrix copy
// * Moving the data from shared memory to Register.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GmemTiledCopy, typename SmemCopyAtom, typename SmemLayout, typename T>
__global__ void matrix_copy_kernel(T const *in, T *out) {
    __shared__ T smem[cosize_v<SmemLayout>];

    GmemTiledCopy tiled_copy;
    auto gA = make_tensor(make_gmem_ptr(in), SmemLayout{});
    auto sA = make_tensor(make_smem_ptr(smem), SmemLayout{});
    auto gC = make_tensor(make_gmem_ptr(out), SmemLayout{});

    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto tAgA = thr_copy.partition_S(gA);
    auto tAsA = thr_copy.partition_D(sA);

    copy(tiled_copy, tAgA, tAsA);
    cp_async_fence();
    cp_async_wait<0>();

    SmemCopyAtom smem_copy_atom;
    auto smem_thr_copy = smem_copy_atom.get_thread_slice(threadIdx.x);
    auto tCsA = smem_thr_copy.partition_S(sA);
    auto tCgc = smem_thr_copy.partition_D(gC);
    auto tCrA = make_tensor<T>(shape(tCgc));
    clear(tCrA);

    // if (thread0()) {
    //     print(tAsA);
    //     print(tCrA);
    //     print(tCsA);
    // }
    copy(smem_copy_atom, tCsA, tCrA);

    transform(tCrA, pre_increment{});

    copy(tCrA, tCgc);
}

void test_matrix_copy() {
    using gmem_tiled_copy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint32_t>, half_t>{},
      Layout<Shape<_8, _4>>{},
      Layout<Shape<_1, _2>>{}));

    using smem_layout = Layout<Shape<_8, Shape<_2,_4>>,Stride<_2,Stride<_1,_16>>>;
    // using smem_layout = Layout<Shape<_8, _8>>;
    using smem_copy_atom = decltype(make_tiled_copy(
      Copy_Atom<SM75_U32x1_LDSM_N, half_t>{}, Layout<Shape<_8, _4>>{}, Layout<Shape<_1, _2>>{}));

    auto in =
      at::arange(
        decltype(size(gmem_tiled_copy::TiledShape_MN{}))::value,
        at::TensorOptions().dtype(at::kHalf))
        .reshape(
          {size<0>(gmem_tiled_copy::TiledShape_MN{}), size<1>(gmem_tiled_copy::TiledShape_MN{})});

    auto out = at::zeros_like(in);

    half_t *d_in, *d_out;
    cudaMalloc((void **)&d_in, in.numel() * in.element_size());
    cudaMalloc((void **)&d_out, out.numel() * out.element_size());

    cudaMemcpy(d_in, in.data_ptr(), in.numel() * in.element_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out.data_ptr(), out.numel() * out.element_size(), cudaMemcpyHostToDevice);

    matrix_copy_kernel<gmem_tiled_copy, smem_copy_atom, smem_layout><<<1, 32>>>(d_in, d_out);
    cudaMemcpy(out.data_ptr(), d_out, out.numel() * out.element_size(), cudaMemcpyDeviceToHost);
    std::cout << out << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    // test_normal_copy();

    test_matrix_copy();
    cudaDeviceReset();

    // test_copy_host();
}