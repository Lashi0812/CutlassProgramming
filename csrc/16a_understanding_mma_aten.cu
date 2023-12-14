#include "cute/pointer.hpp"
#include "cute/swizzle.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/numeric/half.hpp"
#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/ops/rand.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Half.h>


using namespace cute;

// clang-format off
template <
  typename TA             ,  typename TB             ,  typename TC             ,
  typename Layout_GA      ,  typename Layout_GB      ,  typename Layout_GC      ,
  typename Layout_SA      ,  typename Layout_SB      ,
  typename Swi_Layout_SA  ,  typename Swi_Layout_SB  ,
  typename GS_TiledCopy_A ,  typename GS_TiledCopy_B,
  typename SR_TiledCopy_A ,  typename SR_TiledCopy_B,
  typename TiledMMA_>
__global__ void kernel_mma(
  TA             const *A             , TB             const *B             , TC *C,
  Layout_GA             layout_gA     , Layout_GB             layout_gB     , Layout_GC layout_gC,
  Layout_SA             layout_sA     , Layout_SB             layout_sB     ,
  Swi_Layout_SA         swi_layout_sA , Swi_Layout_SB         swi_layout_sB ,
  GS_TiledCopy_A        gs_tiledCopy_A, GS_TiledCopy_B        gs_tiledCopy_B,
  SR_TiledCopy_A        sr_tiledCopy_A, SR_TiledCopy_B        sr_tiledCopy_B,
  TiledMMA_ tiledMAA)
// clang-format on
{
    extern __shared__ char smem[];

    struct SharedStorage {
        cute::array_aligned<TA, cute::cosize_v<Layout_SA>> smem_a;
        cute::array_aligned<TB, cute::cosize_v<Layout_SB>> smem_b;
    };

    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem);

    auto gA = make_tensor(make_gmem_ptr(A), layout_gA);
    auto gB = make_tensor(make_gmem_ptr(B), layout_gB);
    auto gC = make_tensor(make_gmem_ptr(C), layout_gC);

    auto sA = make_tensor(make_smem_ptr(storage.smem_a.data()), swi_layout_sA);
    auto sB = make_tensor(make_smem_ptr(storage.smem_b.data()), layout_sB);

    auto gs_thr_copy_A = gs_tiledCopy_A.get_thread_slice(threadIdx.x);
    auto tAgA = gs_thr_copy_A.partition_S(gA);
    auto tAsA = gs_thr_copy_A.partition_D(sA);

    auto gs_thr_copy_B = gs_tiledCopy_B.get_thread_slice(threadIdx.x);
    auto tBgB = gs_thr_copy_B.partition_S(gB);
    auto tBsB = gs_thr_copy_B.partition_D(sB);

    copy(gs_tiledCopy_A, tAgA, tAsA);
    copy(gs_tiledCopy_B, tBgB, tBsB);
    cp_async_fence();
    cp_async_wait<0>();

    auto thr_mma = tiledMAA.get_thread_slice(threadIdx.x);
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);
    auto tCrC = thr_mma.partition_C(gC);

    auto sr_thr_copy_A = sr_tiledCopy_A.get_thread_slice(threadIdx.x);
    auto tCsA = sr_thr_copy_A.partition_S(sA);
    auto tCrA_view = sr_thr_copy_A.retile_D(tCrA);

    auto sr_thr_copy_B = sr_tiledCopy_B.get_thread_slice(threadIdx.x);
    auto tCsB = sr_thr_copy_B.partition_S(sB);
    auto tCrB_view = sr_thr_copy_B.retile_D(tCrB);

    // auto ptr_as = &tCrA(0);
    // auto ptr_ad = &tCrA_view(0);
    // auto ptr_bs = &tCrB(0);
    // auto ptr_bd = &tCrB_view(0);

    copy(sr_tiledCopy_A, tCsA, tCrA_view);
    copy(sr_tiledCopy_B, tCsB, tCrB_view);

    auto fragC = partition_fragment_C(tiledMAA, shape(gC));
    clear(fragC);
    // auto ptr_cs = &fragC(0);
    // auto ptr_cd = &tCrC(0);

    // if (thread0()) {
    // // clang-format off
    //     print("%%  GA            : ");print(gA                            );print("\n");
    //     print("%%  GB            : ");print(gB                            );print("\n");
    //     print("%%  GC            : ");print(gC                            );print("\n");
    //     print("%%  SA            : ");print(sA                            );print("\n");
    //     print("%%  SB            : ");print(sB                            );print("\n");
    //     print("%%  GS_THR_COPY_A : ");print(gs_thr_copy_A                 );print("\n");
    //     print("%%  TAGA          : ");print(tAgA                          );print("\n");
    //     print("%%  TASA          : ");print(tAsA                          );print("\n");
    //     print("%%  GS_THR_COPY_B : ");print(gs_thr_copy_B                 );print("\n");
    //     print("%%  TBGB          : ");print(tBgB                          );print("\n");
    //     print("%%  TBSB          : ");print(tBsB                          );print("\n");
    //     print("%%  THR_MMA       : ");print(thr_mma                       );print("\n");
    //     print("%%  TCRA          : ");print(tCrA                          );print("\n");
    //     print("%%  TCRB          : ");print(tCrB                          );print("\n");
    //     print("%%  layout_tv_A   : ");print(tiledMAA     .get_layoutA_TV());print("\n");
    //     print("%%  SR_THR_COPY_A : ");print(sr_thr_copy_A                 );print("\n");
    //     print("%%  TCSA          : ");print(tCsA                          );print("\n");
    //     print("%%  TCRA_VIEW     : ");print(tCrA_view                     );print("\n");
    //     print("%%  layout_tv_B   : ");print(tiledMAA     .get_layoutB_TV());print("\n");
    //     print("%%  SR_THR_COPY_B : ");print(sr_thr_copy_B                 );print("\n");
    //     print("%%  TCSB          : ");print(tCsB                          );print("\n");
    //     print("%%  TCRB_VIEW     : ");print(tCrB_view                     );print("\n");
    //     print("%%  FRAGC         : ");print(fragC                         );print("\n");
    //     print("%%  rank TCRA_VIEW: ");print(rank(tCrA_view)                     );print("\n");
    //     print("%%  rank TCRB_VIEW: ");print(rank(tCrB_view)                     );print("\n");
    //     print("%%  rank FragC    : ");print(rank(fragC)                         );print("\n");
    // // clang-format on
    // }

    gemm(tiledMAA, fragC, tCrA_view, tCrB_view, fragC);
    copy(fragC, tCrC);
}

void host_mma() {

    auto layout_gA = Layout<Shape<_16, _16>, Shape<_16, _1>>{};
    auto layout_gB = Layout<Shape<_8, _16>>{};
    auto layout_gC = Layout<Shape<_16, _8>, Shape<_8, _1>>{};

    auto layout_sA = Layout<Shape<_16, _16>, Shape<_16, _1>>{};
    auto layout_sB = Layout<Shape<_8, _16>>{};

    struct SharedStorage {
        cute::array_aligned<half_t, cute::cosize_v<decltype(layout_sA)>> smem_a;
        cute::array_aligned<half_t, cute::cosize_v<decltype(layout_sB)>> smem_b;
    };

    static constexpr int SHARED_SIZE = static_cast<int>(sizeof(SharedStorage));

    auto smem_layout_A = composition(Swizzle<2, 3, 3>{}, layout_sA);
    auto smem_layout_B = composition(Swizzle<2, 2, 4>{}, layout_sB);

    auto tiled_mma = TiledMMA<MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>{};

    auto gsTiledCopyA = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_16, _2>,Stride<_2,_1>>{},
      Layout<Shape<_1, _8>>{});
    auto gsTiledCopyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, half_t>{},
      Layout<Shape<_2, _16>>{},
      Layout<Shape<_4, _1>>{});

    auto srTiledCopyA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, tiled_mma);
    auto srTiledCopyB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, half_t>{}, tiled_mma);

    // clang-format off
    // print("%%  LAYOUT_GA    : ");print(layout_gA   );print("\n");
    // print("%%  LAYOUT_GB    : ");print(layout_gB   );print("\n");
    // print("%%  LAYOUT_GC    : ");print(layout_gC   );print("\n");
    // print("%%  LAYOUT_SA    : ");print(layout_sA   );print("\n");
    // print("%%  LAYOUT_SB    : ");print(layout_sB   );print("\n");
    // print("%%  TILED_MMA    : ");print(tiled_mma   );print("\n");
    // print("%%  GSTILEDCOPYA : ");print(gsTiledCopyA);print("\n");
    // print("%%  GSTILEDCOPYB : ");print(gsTiledCopyB);print("\n");
    // print("%%  SRTILEDCOPYA : ");print(srTiledCopyA);print("\n");
    // print("%%  SRTILEDCOPYB : ");print(srTiledCopyB);print("\n");
    // clang-format on

    // auto [Alayout_MK, Atid] = tiled_mma.get_layoutA_MK();
    // auto [Blayout_NK, Btid] = tiled_mma.get_layoutB_NK();
    // auto [Clayout_MN, Ctid] = tiled_mma.get_layoutC_MN();

    // auto Alayout_TV = tiled_mma.get_layoutA_TV();
    // auto Blayout_TV = tiled_mma.get_layoutB_TV();
    // auto Clayout_TV = tiled_mma.get_layoutC_TV();

    // print_latex_header();
    // std::string test_name("und_mma");
    // mma
    // print_latex(Alayout_MK, (test_name + "_Alayout_MK").c_str());
    // print_latex(Blayout_NK, (test_name + "_Blayout_NK").c_str());
    // print_latex(Clayout_MN, (test_name + "_Clayout_MN").c_str());
    // print_latex(Alayout_TV, (test_name + "_Alayout_TV").c_str());
    // print_latex(Blayout_TV, (test_name + "_Blayout_TV").c_str());
    // print_latex(Clayout_TV, (test_name + "_Clayout_TV").c_str());
    // print_latex(tiled_mma, (test_name + "_tiled_mma").c_str());
    // copy GS
    // auto [gsA_src_MN, gsA_src_MN_thr] = gsTiledCopyA.get_layoutS_MN();
    // auto gsA_src_TV = gsTiledCopyA.get_layoutS_TV();
    // auto [gsA_dst_MN, gsA_dst_MN_thr] = gsTiledCopyA.get_layoutD_MN();
    // auto gsA_dst_TV = gsTiledCopyA.get_layoutD_TV();

    // print_latex(gsA_src_MN, (test_name + "_gsA_colMajor_src_MN").c_str());
    // print_latex(gsA_src_TV, (test_name + "_gsA_colMajor_src_TV").c_str());
    // print_latex(gsA_dst_MN, (test_name + "_gsA_colMajor_dst_MN").c_str());
    // print_latex(gsA_dst_TV, (test_name + "_gsA_colMajor_dst_TV").c_str());
    // print_latex(gsTiledCopyA, (test_name + "_gsTiledCopyA_colmajor").c_str());

    // copy sr
    // auto [srA_src_MN, srA_src_MN_thr] = srTiledCopyA.get_layoutS_MN();
    // auto srA_src_TV = srTiledCopyA.get_layoutS_TV();
    // auto [srA_dst_MN, srA_dst_MN_thr] = srTiledCopyA.get_layoutD_MN();
    // auto srA_dst_TV = srTiledCopyA.get_layoutD_TV();

    // print_latex(srA_src_MN, (test_name + "_srA_colMajor_src_MN").c_str());
    // print_latex(srA_src_TV, (test_name + "_srA_colMajor_src_TV").c_str());
    // print_latex(srA_dst_MN, (test_name + "_srA_colMajor_dst_MN").c_str());
    // print_latex(srA_dst_TV, (test_name + "_srA_colMajor_dst_TV").c_str());
    // print_latex(srTiledCopyA, (test_name + "_srTiledCopyA_colMajor").c_str());
    // print_latex_footer();

    // auto h_A = at::arange(
    //              decltype(size<0>(layout_gA) * size<1>(layout_gA))::value,
    //              at::TensorOptions().dtype(at::kHalf))
    //              .reshape({size<0>(layout_gA), size<1>(layout_gA)});
    // auto h_B = at::arange(
    //              decltype(size<0>(layout_gB) * size<1>(layout_gB))::value,
    //              at::TensorOptions().dtype(at::kHalf))
    //              .reshape({size<0>(layout_gB), size<1>(layout_gB)});
    auto h_A =
      at::rand({size<0>(layout_gA), size<1>(layout_gA)}, at::TensorOptions().dtype(at::kHalf));

    auto h_B =
      at::rand({size<1>(layout_gB), size<0>(layout_gB)}, at::TensorOptions().dtype(at::kHalf));

    auto h_C =
      at::zeros({size<0>(layout_gC), size<1>(layout_gC)}, at::TensorOptions().dtype(at::kHalf));

    half_t *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, h_A.numel() * h_A.element_size());
    cudaMalloc((void **)&d_B, h_B.numel() * h_B.element_size());
    cudaMalloc((void **)&d_C, h_C.numel() * h_C.element_size());

    cudaMemcpy(d_A, h_A.data_ptr(), h_A.numel() * h_A.element_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data_ptr(), h_B.numel() * h_B.element_size(), cudaMemcpyHostToDevice);

    kernel_mma<<<1, 32, SHARED_SIZE>>>(
      d_A,
      d_B,
      d_C,
      layout_gA,
      layout_gB,
      layout_gC,
      layout_sA,
      layout_sB,
      smem_layout_A,
      smem_layout_B,
      gsTiledCopyA,
      gsTiledCopyB,
      srTiledCopyA,
      srTiledCopyB,
      tiled_mma);
    cudaMemcpy(h_C.data_ptr(), d_C, h_C.numel() * h_C.element_size(), cudaMemcpyDeviceToHost);
    // std::cout << (h_A.matmul(h_B).allclose(h_C) ? "MMA Success" : "MMA Failed") << std::endl;
    // std::cout << h_A.matmul(h_B) << std::endl;
    // std::cout << h_C << std::endl;
    // std::cout << (h_A.matmul(h_B).to(at::kHalf) == h_C) << std::endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    host_mma();
    cudaDeviceReset();
    // print_latex(Layout<Shape<_1,_2>>{},"testfdgdfgdfbgfdhfgnjfgnmfgmfghmghmghmghmghmghmghmghm",3);
}