#include "cute/tensor.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/swizzle.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cute/pointer.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/numeric/half.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/util/print.hpp"
#include "latex.hpp"
#include <ATen/ATen.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/zeros_like.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
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

    using smem_layout = Layout<Shape<_8, Shape<_2, _4>>, Stride<_2, Stride<_1, _16>>>;
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//        GS --> Async SR--> Ldmatrix
// 1. A as the row major
// 2. A as the col major
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename OT,
  typename GA_Layout,
  typename SA_Layout,
  typename GS_Tiled_copy,
  typename SR_tiled_copy,
  typename Tiled_MMA_>
__global__ void test_gs_async_sr_ldmatrix_kernel(
  OT const *A,
  OT *out,
  uint32_t *toShareAddr,
  uint32_t *fromShareAddr,
  GA_Layout gA_layout,
  SA_Layout sA_layout,
  GS_Tiled_copy gs_tiled_copy,
  SR_tiled_copy sr_tiled_copy,
  Tiled_MMA_ tiled_mma) {

    __shared__ OT smem_A[cosize_v<SA_Layout>];

    auto gA = make_tensor(make_gmem_ptr(A), gA_layout);
    auto gOut = make_tensor(make_gmem_ptr(out), gA_layout);
    auto sA = make_tensor(make_smem_ptr(smem_A), sA_layout);

    auto sToShareAddr = make_tensor(make_gmem_ptr(toShareAddr), sA_layout);
    auto sFromShareAddr = make_tensor(make_gmem_ptr(fromShareAddr), sA_layout);

    auto gs_thr_copy = gs_tiled_copy.get_thread_slice(threadIdx.x);
    auto tAgA = gs_thr_copy.partition_S(gA);
    auto tAsA = gs_thr_copy.partition_D(sA);
    auto tAgOut = gs_thr_copy.partition_S(gOut);

    auto tAsTo = gs_thr_copy.partition_D(sToShareAddr);
    for (int i = 0; i < size(tAsA); ++i) {
        tAsTo(i) = cast_smem_ptr_to_uint(&tAsA(i));
    }

    copy(gs_tiled_copy, tAgA, tAsA);
    cp_async_fence();
    cp_async_wait<0>();

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto tCrA = thr_mma.partition_fragment_A(sA);

    auto sr_thr_copy = sr_tiled_copy.get_thread_slice(threadIdx.x);
    auto tCsA = sr_thr_copy.partition_S(sA);
    auto tCrA_view = sr_thr_copy.retile_D(tCrA);

    auto tCsFrom = sr_thr_copy.partition_S(sFromShareAddr);
    for (int i = 0; i < size(tCsA); ++i) {
        tCsFrom(i) = cast_smem_ptr_to_uint(&tCsA(i));
    }

    auto ptr1 = &(tCrA_view(0));
    auto ptr2 = &(tCsA(0));
    copy(sr_tiled_copy, tCsA, tCrA_view);
    transform(tCrA_view, pre_increment{});
    copy(tCrA_view, tAgOut);
}

template <
  typename GS_WT = uint128_t,
  typename OT = half_t,
  typename GA_Layout,
  typename SA_Layout,
  typename GS_ThrLayout,
  typename GS_ValLayout,
  typename MMA_ATOM_OP = SM80_16x8x16_F16F16F16F16_TN,
  typename SR_CP_OP = SM75_U32x4_LDSM_N>
void test_gs_async_sr_ldmatrix_host(
  std::string test_name,
  OT const *A,
  OT *gOut,
  uint32_t *toShareAddr,
  uint32_t *fromShareAddr,
  GA_Layout,
  SA_Layout,
  GS_ThrLayout,
  GS_ValLayout,
  MMA_ATOM_OP,
  SR_CP_OP) {
    auto gA_layout = GA_Layout{};
    auto sA_layout = SA_Layout{};
    auto gs_tiled_copy = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<GS_WT>, OT>{}, GS_ThrLayout{}, GS_ValLayout{});

    auto tiled_mma = TiledMMA<MMA_Atom<MMA_ATOM_OP>>{};
    auto sr_tiled_copy = make_tiled_copy_A(Copy_Atom<SR_CP_OP, OT>{}, tiled_mma);

    if (0) {
        print_latex_header();
        // clang-format off
        print("%%  GA_LAYOUT     : ");print_latex(gA_layout     ,(test_name+"_GA_LAYOUT"    ).c_str());print("\n");
        print("%%  SA_LAYOUT     : ");print_latex(sA_layout     ,(test_name+"_SA_LAYOUT"    ).c_str());print("\n");
        print("%%  GS_TILED_COPY : ");print_latex(gs_tiled_copy ,(test_name+"_GS_TILED_COPY").c_str());print("\n");
        print("%%  TILED_MMA     : ");print_latex(tiled_mma     ,(test_name+"_TILED_MMA"    ).c_str());print("\n");
        print("%%  SR_TILED_COPY : ");print_latex(sr_tiled_copy ,(test_name+"_SR_TILED_COPY").c_str());print("\n");
        // clang-format on

        // copy-gs
        auto [gsA_src_MN, gsA_src_MN_thr] = gs_tiled_copy.get_layoutS_MN();
        auto gsA_src_TV = gs_tiled_copy.get_layoutS_TV();
        auto [gsA_dst_MN, gsA_dst_MN_thr] = gs_tiled_copy.get_layoutD_MN();
        auto gsA_dst_TV = gs_tiled_copy.get_layoutD_TV();

        print_latex(gsA_src_MN, (test_name + "_gsA_src_MN").c_str());
        print_latex(gsA_src_TV, (test_name + "_gsA_src_TV").c_str());
        print_latex(gsA_dst_MN, (test_name + "_gsA_dst_MN").c_str());
        print_latex(gsA_dst_TV, (test_name + "_gsA_dst_TV").c_str());

        // copy-sr
        auto [srA_src_MN, srA_src_MN_thr] = sr_tiled_copy.get_layoutS_MN();
        auto srA_src_TV = sr_tiled_copy.get_layoutS_TV();
        auto [srA_dst_MN, srA_dst_MN_thr] = sr_tiled_copy.get_layoutD_MN();
        auto srA_dst_TV = sr_tiled_copy.get_layoutD_TV();

        print_latex(srA_src_MN, (test_name + "_srA_src_MN").c_str());
        print_latex(srA_src_TV, (test_name + "_srA_src_TV").c_str());
        print_latex(srA_dst_MN, (test_name + "_srA_dst_MN").c_str());
        print_latex(srA_dst_TV, (test_name + "_srA_dst_TV").c_str());

        print_latex_footer();
    }
    // clang-foramt on

    // kernel
    test_gs_async_sr_ldmatrix_kernel<<<1, 32>>>(
      A,
      gOut,
      toShareAddr,
      fromShareAddr,
      gA_layout,
      sA_layout,
      gs_tiled_copy,
      sr_tiled_copy,
      tiled_mma);
}

void test_gs_async_sr_ldmatrix_examples() {
    // test 1 --> row major
    // {
    //     auto gA_layout = Layout<Shape<_16, _16>>{};
    //     auto sA_layout = Layout<Shape<_16, _16>>{};
    //     auto thr_layout = Layout<Shape<_2, _16>>{};
    //     auto val_layout = Layout<Shape<_8, _1>>{};
    //     auto mma_atom_op = SM80_16x8x16_F16F16F16F16_TN{};
    //     auto sr_cp_op = SM75_U32x4_LDSM_N{};

    //     auto h_A = at::arange(
    //                  decltype(size<0>(gA_layout) * size<1>(gA_layout))::value,
    //                  at::TensorOptions().dtype(at::kHalf))
    //                  .reshape({size<0>(gA_layout), size<1>(gA_layout)});
    //     half_t *d_A;
    //     cudaMalloc((void **)&d_A, h_A.numel() * h_A.element_size());
    //     cudaMemcpy(d_A, h_A.data_ptr(), h_A.numel() * h_A.element_size(),
    //     cudaMemcpyHostToDevice);

    //     test_gs_async_sr_ldmatrix_host(
    //       "gs_async_sr_ldmatrix_row",
    //       d_A,
    //       gA_layout,
    //       sA_layout,
    //       thr_layout,
    //       val_layout,
    //       mma_atom_op,
    //       sr_cp_op);
    // }

    // test 2 --> col major
    {
        auto gA_layout = Layout<Shape<_16, _16>, Stride<_16, _1>>{};
        auto sA_layout = Layout<Shape<_16, _16>, Stride<_16, _1>>{};
        auto thr_layout = Layout<Shape<_16, _2>>{};
        auto val_layout = Layout<Shape<_1, _8>>{};
        auto mma_atom_op = SM80_16x8x16_F16F16F16F16_TN{};
        auto sr_cp_op = SM75_U32x4_LDSM_N{};

        auto h_A = at::arange(
                     decltype(size<0>(gA_layout) * size<1>(gA_layout))::value,
                     at::TensorOptions().dtype(at::kHalf))
                     .reshape({size<0>(gA_layout), size<1>(gA_layout)});
        auto h_out = at::zeros_like(h_A);

        auto h_toShareAddr = at::arange(
                               decltype(size<0>(gA_layout) * size<1>(gA_layout))::value,
                               at::TensorOptions().dtype(at::kInt))
                               .reshape({size<0>(gA_layout), size<1>(gA_layout)});
        auto h_fromShareAddr = at::arange(
                                 decltype(size<0>(gA_layout) * size<1>(gA_layout))::value,
                                 at::TensorOptions().dtype(at::kInt))
                                 .reshape({size<0>(gA_layout), size<1>(gA_layout)});

        half_t *d_A, *d_out;
        uint32_t *d_toShareAddr, *d_fromShareAddr;
        cudaMalloc((void **)&d_A, h_A.numel() * h_A.element_size());
        cudaMalloc((void **)&d_out, h_out.numel() * h_out.element_size());
        cudaMalloc((void **)&d_toShareAddr, h_toShareAddr.numel() * h_toShareAddr.element_size());
        cudaMalloc(
          (void **)&d_fromShareAddr, h_fromShareAddr.numel() * h_fromShareAddr.element_size());
        cudaMemcpy(d_A, h_A.data_ptr(), h_A.numel() * h_A.element_size(), cudaMemcpyHostToDevice);

        test_gs_async_sr_ldmatrix_host(
          "gs_async_sr_ldmatrix_col",
          d_A,
          d_out,
          d_toShareAddr,
          d_fromShareAddr,
          gA_layout,
          sA_layout,
          thr_layout,
          val_layout,
          mma_atom_op,
          sr_cp_op);

        cudaMemcpy(
          h_out.data_ptr(), d_out, h_A.numel() * h_A.element_size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(
          h_toShareAddr.data_ptr(), d_toShareAddr, h_toShareAddr.numel() * h_toShareAddr.element_size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(
          h_fromShareAddr.data_ptr(), d_fromShareAddr, h_fromShareAddr.numel() * h_fromShareAddr.element_size(), cudaMemcpyDeviceToHost);
        std::cout << h_out << std::endl;
        std::cout << h_toShareAddr << std::endl;
        std::cout << h_fromShareAddr << std::endl;
    }
}

int main() {
    // test_normal_copy();

    // test_matrix_copy();
    // test_copy_host();

    test_gs_async_sr_ldmatrix_examples();

    cudaDeviceReset();
}