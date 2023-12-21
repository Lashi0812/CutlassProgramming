#include "cute/tensor.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
#include "cute/swizzle.hpp"
#include "cute/util/type_traits.hpp"
#include <ATen/ATen.h>

using namespace cute;

// utility

template <int I, class Shape>
using get_t = decltype(get<I>(declval<Shape>()));

template <int I, class Shape>
static constexpr int get_v = get_t<I, Shape>::value;

struct KMajor {};
struct MMajor {};

template <
  typename TA,
  typename TB,
  typename TC,
  typename Layout_GA,
  typename Layout_GB,
  typename Layout_GC,
  typename Layout_SA,
  typename Layout_SB,
  typename Layout_SA_After,
  typename Layout_SB_After,
  typename GS_TiledCopy_A,
  typename GS_TiledCopy_B,
  typename SR_TiledCopy_A,
  typename SR_TiledCopy_B,
  typename TiledMMA_>
__global__ void kernel_mma(
  TA const       *A,
  TB const       *B,
  TC             *C,
  Layout_GA       layout_gA,
  Layout_GB       layout_gB,
  Layout_GC       layout_gC,
  Layout_SA       layout_sA,
  Layout_SB       layout_sB,
  Layout_SA_After layout_sA_after,
  Layout_SB_After layout_sB_after,
  GS_TiledCopy_A  gs_tiledCopy_A,
  GS_TiledCopy_B  gs_tiledCopy_B,
  SR_TiledCopy_A  sr_tiledCopy_A,
  SR_TiledCopy_B  sr_tiledCopy_B,
  TiledMMA_       tiledMAA)
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
    auto tAgA          = gs_thr_copy_A.partition_S(gA);
    auto tAsA          = gs_thr_copy_A.partition_D(sA);

    auto gs_thr_copy_B = gs_tiledCopy_B.get_thread_slice(threadIdx.x);
    auto tBgB          = gs_thr_copy_B.partition_S(gB);
    auto tBsB          = gs_thr_copy_B.partition_D(sB);

    copy(gs_tiledCopy_A, tAgA, tAsA);
    copy(gs_tiledCopy_B, tBgB, tBsB);
    cp_async_fence();
    cp_async_wait<0>();

    auto thr_mma = tiledMAA.get_thread_slice(threadIdx.x);
    auto tCrA    = thr_mma.partition_fragment_A(sA);
    auto tCrB    = thr_mma.partition_fragment_B(sB);
    auto tCrC    = thr_mma.partition_C(gC);

    auto sA_after      = make_tensor(make_smem_ptr(storage.smem_a.data()), layout_sA_after);
    auto sr_thr_copy_A = sr_tiledCopy_A.get_thread_slice(threadIdx.x);
    auto tCsA          = sr_thr_copy_A.partition_S(sA_after);
    auto tCrA_view     = sr_thr_copy_A.retile_D(tCrA);

    auto sB_after      = make_tensor(make_smem_ptr(storage.smem_b.data()), layout_sB_after);
    auto sr_thr_copy_B = sr_tiledCopy_B.get_thread_slice(threadIdx.x);
    auto tCsB          = sr_thr_copy_B.partition_S(sB_after);
    auto tCrB_view     = sr_thr_copy_B.retile_D(tCrB);

    // auto ptr1 = &(tCrA(0));
    copy(sr_tiledCopy_A, tCsA, tCrA_view);
    copy(sr_tiledCopy_B, tCsB, tCrB_view);

    auto fragC = partition_fragment_C(tiledMAA, shape(gC));
    clear(fragC);

    gemm(tiledMAA, fragC, tCrA, tCrB, fragC);
    copy(fragC, tCrC);
}

template <typename GEMM_Config_>
void host_mma(std::string test_case) {

    using AConfig  = typename GEMM_Config_::OperandA_;
    using BConfig  = typename GEMM_Config_::OperandB_;
    using ShapeMNK = typename GEMM_Config_::ProblemShapeMNK;

    auto layout_gA = typename AConfig::GmemLayout{};
    auto layout_gB = typename BConfig::GmemLayout{};
    auto layout_gC =
      Layout<Shape<get_t<0, ShapeMNK>, get_t<1, ShapeMNK>>, Shape<get_t<1, ShapeMNK>, _1>>{};

    auto layout_sA = typename AConfig::SmemLayout{};
    auto layout_sB = typename BConfig::SmemLayout{};

    auto layout_sA_after = typename AConfig::SmemLayout{};
    auto layout_sB_after = typename BConfig::SmemLayout{};

    struct SharedStorage {
        cute::array_aligned<tfloat32_t, cute::cosize_v<decltype(layout_sA)>> smem_a;
        cute::array_aligned<tfloat32_t, cute::cosize_v<decltype(layout_sB)>> smem_b;
    };

    static constexpr int SHARED_SIZE = static_cast<int>(sizeof(SharedStorage));

    // auto smem_layout_A = composition(Swizzle<2, 3, 3>{}, layout_sA);
    // auto smem_layout_B = composition(Swizzle<2, 2, 4>{}, layout_sB);

    auto tiled_mma = typename GEMM_Config_::TiledMMA_{};

    auto gsTiledCopyA = typename AConfig::GSTiledCopy{};
    auto gsTiledCopyB = typename BConfig::GSTiledCopy{};

    auto srTiledCopyA = make_tiled_copy_A(typename AConfig::SRCopyAtom{}, tiled_mma);
    auto srTiledCopyB = make_tiled_copy_B(typename BConfig::SRCopyAtom{}, tiled_mma);

    // auto h_A = at::arange(
    //              decltype(get<0>(layout_gA) * get<1>(layout_gA))::value,
    //              at::TensorOptions().dtype(at::kFloat))
    //              .reshape({get<0>(layout_gA), get<1>(layout_gA)});
    // auto h_B = at::arange(
    //              decltype(get<0>(layout_gB) * get<1>(layout_gB))::value,
    //              at::TensorOptions().dtype(at::kFloat))
    //              .reshape({get<0>(layout_gB), get<1>(layout_gB)});

    auto h_A =
      at::rand({size<0>(layout_gA), size<1>(layout_gA)}, at::TensorOptions().dtype(at::kFloat));

    auto h_B =
      at::rand({size<0>(layout_gB), size<1>(layout_gB)}, at::TensorOptions().dtype(at::kFloat));

    auto h_C =
      at::zeros({size<0>(layout_gC), size<1>(layout_gC)}, at::TensorOptions().dtype(at::kFloat));

    tfloat32_t *d_A, *d_B;
    float      *d_C;
    cudaMalloc((void **)&d_A, h_A.numel() * h_A.element_size());
    cudaMalloc((void **)&d_B, h_B.numel() * h_B.element_size());
    cudaMalloc((void **)&d_C, h_C.numel() * h_C.element_size());

    cudaMemcpy(d_A, h_A.data_ptr(), h_A.numel() * h_A.element_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data_ptr(), h_B.numel() * h_B.element_size(), cudaMemcpyHostToDevice);

    kernel_mma<<<1, 128, SHARED_SIZE>>>(
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
        cpu_ans = (h_A.matmul(h_B));
    else if (test_case == "NT")
        cpu_ans = (h_A.matmul(h_B.mT()));
    else if (test_case == "TN")
        cpu_ans = (h_A.mT().matmul(h_B));
    else if (test_case == "TT")
        cpu_ans = (h_A.mT().matmul(h_B.mT()));

    std::cout << (h_C.allclose(cpu_ans, 1e-02) ? "MMA Success" : "MMA Failed") << std::endl;
    // std::cout << cpu_ans << std::endl;
    // std::cout << h_C << std::endl;
    // std::cout << (h_C.isclose(cpu_ans, 1e-02)) << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Operand A --> MxK --> Based on the Stride we define is K Major or M Major
// * Stride<X,1> --> K Major
// * Stride<1,X> --> M Major

template <typename ShapeMK, typename Major>
struct OperandA;

template <typename ShapeNK, typename Major>
struct OperandB;

template <typename ShapeMK>
struct OperandA<ShapeMK, KMajor> {
    using GmemLayout =
      Layout<Shape<get_t<0, ShapeMK>, get_t<1, ShapeMK>>, Stride<get_t<1, ShapeMK>, _1>>;

    using SmemLayout = decltype(composition(
      Swizzle<1, 2, 3>{},
      Layout<Shape<get_t<0, ShapeMK>, get_t<1, ShapeMK>>, Stride<get_t<1, ShapeMK>, _1>>{}));

    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, tfloat32_t>{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _2>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, tfloat32_t>;
};

template <typename ShapeMK>
struct OperandA<ShapeMK, MMajor> {
    using GmemLayout =
      Layout<Shape<get_t<0, ShapeMK>, get_t<1, ShapeMK>>, Stride<_1, get_t<0, ShapeMK>>>;

    using SmemLayout = decltype(composition(
      Swizzle<1, 2, 3>{},
      Layout<Shape<get_t<0, ShapeMK>, get_t<1, ShapeMK>>, Stride<_1, get_t<0, ShapeMK>>>{}));

    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, tfloat32_t>{},
      Layout<Shape<_16, _8>, Stride<_1, _16>>{},
      Layout<Shape<_2, _1>>{}));

    using SRCopyAtom = Copy_Atom<SM75_U16x8_LDSM_T, tfloat32_t>;
};

template <typename ShapeNK>
struct OperandB<ShapeNK, KMajor> : OperandA<ShapeNK, KMajor> {};

template <typename ShapeNK>
struct OperandB<ShapeNK, MMajor> : OperandA<ShapeNK, MMajor> {};

struct GEMM_Config {
    using TiledMMA_ = TiledMMA<
      MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
      Layout<Shape<_2, _2, _1>>,
      Layout<Shape<_1, _2, _1>>>;
    using ProblemShapeMNK            = TiledMMA_::TiledShape_MNK;
    using ProblemShapeMK             = Shape<get_t<0, ProblemShapeMNK>, get_t<2, ProblemShapeMNK>>;
    using ProblemShapeNK             = Shape<get_t<1, ProblemShapeMNK>, get_t<2, ProblemShapeMNK>>;
    static constexpr int ThreadCount = 128;
    using OperandA_                  = OperandA<ProblemShapeMK, KMajor>;
    using OperandB_                  = OperandB<ProblemShapeNK, KMajor>;
};

void host_mma_examples() { host_mma<GEMM_Config>("NT"); }

int main() {
    host_mma_examples();
    cudaDeviceReset();
}