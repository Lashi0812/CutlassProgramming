#include "cute/pointer.hpp"
#include "cute/swizzle.hpp"
#include "cute/swizzle_layout.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/numeric/half.hpp"
#include "cutlass/half.h"
#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/zeros_like.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Half.h>
#include <cstdint>

using namespace cute;

// clang-format off
template <
  typename TA              , typename TB        , typename TC ,
  typename Layout_GA       , typename Layout_GB , typename Layout_GC ,
  typename Layout_SA       , typename Layout_SB ,
  typename Layout_SA_After , typename Layout_SB_After ,
  typename GS_TiledCopy_A  , typename GS_TiledCopy_B ,
  typename SR_TiledCopy_A  , typename SR_TiledCopy_B ,
  typename TiledMMA_       >
__global__ void kernel_mma(
  TA              const *A               , TB              const *B               , TC *C,
  Layout_GA              layout_gA       , Layout_GB              layout_gB       , Layout_GC layout_gC,
  Layout_SA              layout_sA       , Layout_SB              layout_sB       ,
  Layout_SA_After        layout_sA_after , Layout_SB_After        layout_sB_after ,
  GS_TiledCopy_A         gs_tiledCopy_A  , GS_TiledCopy_B         gs_tiledCopy_B  ,
  SR_TiledCopy_A         sr_tiledCopy_A  , SR_TiledCopy_B         sr_tiledCopy_B  ,
  TiledMMA_              tiledMAA       )
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

    auto sA_after = make_tensor(make_smem_ptr(storage.smem_a.data()), layout_sA_after);
    auto sr_thr_copy_A = sr_tiledCopy_A.get_thread_slice(threadIdx.x);
    auto tCsA = sr_thr_copy_A.partition_S(sA_after);
    auto tCrA_view = sr_thr_copy_A.retile_D(tCrA);

    auto sB_after = make_tensor(make_smem_ptr(storage.smem_b.data()), layout_sB_after);
    auto sr_thr_copy_B = sr_tiledCopy_B.get_thread_slice(threadIdx.x);
    auto tCsB = sr_thr_copy_B.partition_S(sB_after);
    auto tCrB_view = sr_thr_copy_B.retile_D(tCrB);

    copy(sr_tiledCopy_A, tCsA, tCrA_view);
    // auto ptr1 = &(tCrB_view(0));
    copy(sr_tiledCopy_B, tCsB, tCrB_view);

    auto fragC = partition_fragment_C(tiledMAA, shape(gC));
    clear(fragC);

    gemm(tiledMAA, fragC, tCrA, tCrB, fragC);
    copy(fragC, tCrC);
}

template <typename AConfig, typename BConfig, typename AShape, typename BShape>
void host_mma(std::string test_case) {

    auto layout_gA = typename AConfig::GmemLayout{};
    auto layout_gB = typename BConfig::GmemLayout{};
    auto layout_gC = Layout<Shape<_16, _8>, Shape<_8, _1>>{};

    auto layout_sA = typename AConfig::SmemLayout{};
    auto layout_sB = typename BConfig::SmemLayout{};

    auto layout_sA_after = typename AConfig::SmemLayoutAfter{};
    auto layout_sB_after = typename BConfig::SmemLayoutAfter{};

    struct SharedStorage {
        cute::array_aligned<half_t, cute::cosize_v<decltype(layout_sA)>> smem_a;
        cute::array_aligned<half_t, cute::cosize_v<decltype(layout_sB)>> smem_b;
    };

    static constexpr int SHARED_SIZE = static_cast<int>(sizeof(SharedStorage));

    // auto smem_layout_A = composition(Swizzle<2, 3, 3>{}, layout_sA);
    // auto smem_layout_B = composition(Swizzle<2, 2, 4>{}, layout_sB);

    auto tiled_mma = TiledMMA<MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>{};

    auto gsTiledCopyA = typename AConfig::GSTiledCopy{};
    auto gsTiledCopyB = typename BConfig::GSTiledCopy{};

    auto srTiledCopyA = make_tiled_copy_A(typename AConfig::SRCopyAtom{}, tiled_mma);
    auto srTiledCopyB = make_tiled_copy_B(typename BConfig::SRCopyAtom{}, tiled_mma);

    // auto h_A = at::arange(
    //              decltype(size<0>(AShape{}) * size<1>(AShape{}))::value,
    //              at::TensorOptions().dtype(at::kHalf))
    //              .reshape({size<0>(AShape{}), size<1>(AShape{})});
    // auto h_B = at::arange(
    //              decltype(size<0>(BShape{}) * size<1>(BShape{}))::value,
    //              at::TensorOptions().dtype(at::kHalf))
    //              .reshape({size<0>(BShape{}), size<1>(BShape{})});

    auto h_A =
      at::rand({size<0>(AShape{}), size<1>(AShape{})}, at::TensorOptions().dtype(at::kHalf));

    auto h_B =
      at::rand({size<0>(BShape{}), size<1>(BShape{})}, at::TensorOptions().dtype(at::kHalf));

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
      layout_sA_after,
      layout_sB_after,
      gsTiledCopyA,
      gsTiledCopyB,
      srTiledCopyA,
      srTiledCopyB,
      tiled_mma);
    cudaMemcpy(h_C.data_ptr(), d_C, h_C.numel() * h_C.element_size(), cudaMemcpyDeviceToHost);

    auto cpu_ans = at::zeros_like(h_C);
    if (test_case == "NN")
        cpu_ans = h_A.matmul(h_B);
    else if (test_case == "NT")
        cpu_ans = h_A.matmul(h_B.mT());
    else if (test_case == "TN")
        cpu_ans = h_A.mT().matmul(h_B);
    else if (test_case == "TT")
        cpu_ans = h_A.mT().matmul(h_B.mT());

    std::cout << (h_C.allclose(cpu_ans, 1e-02) ? "MMA Success" : "MMA Failed") << std::endl;
    std::cout << cpu_ans << std::endl;
    std::cout << h_C << std::endl;
    std::cout << (cpu_ans.to(at::kHalf) == h_C) << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                        GEMM Configuration
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct A_ShapeMxK_Mmajor {
    using GmemLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
    using SmemLayout =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_16, _16>, Stride<_1, _16>>{}));
    using SmemLayoutAfter =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_16, _16>, Stride<_16, _1>>{}));
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_2, _16>, Stride<_1, _2>>{},
      Layout<Shape<_8, _1>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
};

struct A_ShapeMxK_Kmajor {
    using GmemLayout = Layout<Shape<_16, _16>, Stride<_16, _1>>;
    using SmemLayout =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_16, _16>, Stride<_16, _1>>{}));
    using SmemLayoutAfter =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_16, _16>, Stride<_16, _1>>{}));
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_16, _2>, Stride<_2, _1>>{},
      Layout<Shape<_1, _8>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
};

struct A_ShapeKxM_Mmajor {
    using GmemLayout = Layout<Shape<_16, _16>, Stride<_1, _16>>;
    using SmemLayout =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_16, _16>, Stride<_1, _16>>{}));
    using SmemLayoutAfter =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_16, _16>, Stride<_1, _16>>{}));
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_2, _16>, Stride<_1, _2>>{},
      Layout<Shape<_8, _1>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;
};

struct A_ShapeKxM_Kmajor {
    using GmemLayout = Layout<Shape<_16, _16>, Stride<_16, _1>>;
    using SmemLayout =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_16, _16>, Stride<_16, _1>>{}));
    using SmemLayoutAfter =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_16, _16>, Stride<_1, _16>>{}));
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_16, _2>, Stride<_2, _1>>{},
      Layout<Shape<_1, _8>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;
};

struct B_ShapeNxK_Kmajor {
    using GmemLayout = Layout<Shape<_8, _16>, Stride<_16, _1>>;
    using SmemLayout =
      decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
    using SmemLayoutAfter =
      decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, half_t>{},
      Layout<Shape<_8, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _4>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U32x2_LDSM_N, half_t>;
};

struct B_ShapeNxK_Nmajor {
    using GmemLayout = Layout<Shape<_8, _16>, Stride<_1, _8>>;
    using SmemLayout =
      decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_1, _8>>{}));
    using SmemLayoutAfter =
      decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, half_t>{},
      Layout<Shape<_2, _16>, Stride<_1, _2>>{},
      Layout<Shape<_4, _1>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U32x2_LDSM_N, half_t>;
};

struct B_ShapeKxN_Kmajor {
    using GmemLayout = Layout<Shape<_8, _16>, Stride<_16, _1>>;
    using SmemLayout =
      decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
        using SmemLayoutAfter =
      decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_1, _8>>{}));
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, half_t>{},
      Layout<Shape<_8, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _4>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U16x4_LDSM_T, half_t>;
    };

struct B_ShapeKxN_Nmajor {
    using GmemLayout = Layout<Shape<_8, _16>, Stride<_1, _8>>;
    using SmemLayout =
      decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_1, _8>>{}));
    using SmemLayoutAfter =
      decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_1, _8>>{}));
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, half_t>{},
      Layout<Shape<_2, _16>, Stride<_1, _2>>{},
      Layout<Shape<_4, _1>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U16x4_LDSM_T, half_t>;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                        End GEMM Configuration
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void host_mma_examples() {
    host_mma<A_ShapeMxK_Kmajor, B_ShapeNxK_Kmajor, Shape<_16, _16>, Shape<_8, _16>>("NT");
    host_mma<A_ShapeMxK_Kmajor, B_ShapeNxK_Nmajor, Shape<_16, _16>, Shape<_8, _16>>("NT");
    host_mma<A_ShapeMxK_Mmajor, B_ShapeNxK_Kmajor, Shape<_16, _16>, Shape<_8, _16>>("NT");
    host_mma<A_ShapeMxK_Mmajor, B_ShapeNxK_Nmajor, Shape<_16, _16>, Shape<_8, _16>>("NT");

    host_mma<A_ShapeMxK_Kmajor, B_ShapeKxN_Kmajor, Shape<_16, _16>, Shape<_16, _8>>("NN");
    host_mma<A_ShapeMxK_Kmajor, B_ShapeKxN_Nmajor, Shape<_16, _16>, Shape<_16, _8>>("NN");
    host_mma<A_ShapeMxK_Mmajor, B_ShapeKxN_Kmajor, Shape<_16, _16>, Shape<_16, _8>>("NN");
    host_mma<A_ShapeMxK_Mmajor, B_ShapeKxN_Nmajor, Shape<_16, _16>, Shape<_16, _8>>("NN");

    host_mma<A_ShapeKxM_Kmajor, B_ShapeKxN_Kmajor, Shape<_16, _16>, Shape<_16, _8>>("TN");
    host_mma<A_ShapeKxM_Kmajor, B_ShapeKxN_Nmajor, Shape<_16, _16>, Shape<_16, _8>>("TN");
    host_mma<A_ShapeKxM_Mmajor, B_ShapeKxN_Kmajor, Shape<_16, _16>, Shape<_16, _8>>("TN");
    host_mma<A_ShapeKxM_Mmajor, B_ShapeKxN_Nmajor, Shape<_16, _16>, Shape<_16, _8>>("TN");

    host_mma<A_ShapeKxM_Kmajor, B_ShapeNxK_Kmajor, Shape<_16, _16>, Shape<_8, _16>>("TT");
    host_mma<A_ShapeKxM_Kmajor, B_ShapeNxK_Nmajor, Shape<_16, _16>, Shape<_8, _16>>("TT");
    host_mma<A_ShapeKxM_Mmajor, B_ShapeNxK_Kmajor, Shape<_16, _16>, Shape<_8, _16>>("TT");
    host_mma<A_ShapeKxM_Mmajor, B_ShapeNxK_Nmajor, Shape<_16, _16>, Shape<_8, _16>>("TT");
}

int main() {
    host_mma_examples();
    cudaDeviceReset();
}