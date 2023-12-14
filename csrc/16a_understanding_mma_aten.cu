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

    auto sA = make_tensor(make_smem_ptr(storage.smem_a.data()), layout_sA);
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

    copy(sr_tiledCopy_A, tCsA, tCrA_view);
    copy(sr_tiledCopy_B, tCsB, tCrB_view);

    auto fragC = partition_fragment_C(tiledMAA, shape(gC));
    clear(fragC);

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
      Layout<Shape<_16, _2>>{},
      Layout<Shape<_1, _8>>{});
    auto gsTiledCopyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, half_t>{},
      Layout<Shape<_2, _16>>{},
      Layout<Shape<_4, _1>>{});

    auto srTiledCopyA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, tiled_mma);
    auto srTiledCopyB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, half_t>{}, tiled_mma);

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
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    host_mma();
    cudaDeviceReset();
}