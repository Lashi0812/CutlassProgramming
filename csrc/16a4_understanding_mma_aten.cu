#include "cute/stride.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/tfloat.hpp"
#include <ATen/ATen.h>

using namespace cute;

constexpr int exponent(int result) {
    int exponent = 0;
    while (result > 1) {
        result = result >> 1; // Right shift the result by 1 bit
        exponent++;
    }
    return exponent;
}

template <int I, class Shape>
using get_t = decltype(get<I>(declval<Shape>()));

template <typename T, typename sizeBK, typename Major, int ThrCount>
struct OperandA;

template <typename T, typename sizeBK, typename Major, int ThrCount>
struct OperandB;

// 128x128xBK K-Major (Row Major)
template <typename T, typename sizeBK, int ThrCount>
struct OperandA<T, sizeBK, GenRowMajor, ThrCount> {
    static int constexpr M         = exponent(16 / sizeof(T));
    static int constexpr S         = 7 - (exponent(sizeof(T)) + M);
    static int constexpr B         = (3 + exponent(sizeBK::value)) - (7 - exponent(sizeof(T)));
    static int constexpr Alignment = sizeof(uint128_t) / sizeof(T);
    static int constexpr ThrMode1  = sizeBK::value / Alignment;
    static int constexpr ThrMode0  = ThrCount / ThrMode1;

    using SmemLayout =
      decltype(composition(Swizzle<B, M, S>{}, Layout<Shape<_8, sizeBK>, Stride<sizeBK, _1>>{}));
    using SRCopyAtom  = Copy_Atom<SM75_U32x4_LDSM_N, T>;
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T>{},
      Layout<Shape<Int<ThrMode0>, Int<ThrMode1>>, Stride<Int<ThrMode1>, _1>>{},
      Layout<Shape<_1, Int<Alignment>>>{}));
};

// 128x128xBK M-Major (Col Major)
template <typename T, typename sizeBK, int ThrCount>
struct OperandA<T, sizeBK, GenColMajor, ThrCount> {
    static int constexpr M         = exponent(16 / sizeof(T));
    static int constexpr S         = 7 - (exponent(sizeof(T)) + M);
    static int constexpr B         = (3 + exponent(sizeBK::value)) - (7 - exponent(sizeof(T)));
    static int constexpr Alignment = sizeof(uint128_t) / sizeof(T);
    static int constexpr ThrMode1  = sizeBK::value / Alignment;
    static int constexpr ThrMode0  = ThrCount / ThrMode1;

    using SmemLayout =
      decltype(composition(Swizzle<B, M, S>{}, Layout<Shape<sizeBK, _8>, Stride<_1, sizeBK>>{}));
    using SRCopyAtom  = Copy_Atom<SM75_U16x8_LDSM_T, T>;
    using GSTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T>{},
      Layout<Shape<Int<ThrMode0>, Int<ThrMode1>>, Stride<_1, Int<ThrMode0>>>{},
      Layout<Shape<Int<Alignment>, _1>>{}));
};

// Operand B - Column-Major (K-major)
template <typename T, typename sizeBK, int ThrCount>
struct OperandB<T, sizeBK, GenColMajor, ThrCount> : OperandA<T, sizeBK, GenRowMajor, ThrCount> {};

// Operand B - Row-Major (N-major)
template <typename T, typename sizeBK, int ThrCount>
struct OperandB<T, sizeBK, GenRowMajor, ThrCount> : OperandA<T, sizeBK, GenColMajor, ThrCount> {};

template <typename T, typename sizeBK, typename MMA_Op_, typename AMajor, typename BMajor>
struct GemmConfig {
    using TileShape               = Shape<_128, _128, sizeBK>;
    static int constexpr ThrCount = 128;
    using LdMatrixElemShapeMNK    = Shape<_16, _16, Int<(16 * 2) / sizeof(T)>>;
    using ValShape                = decltype(transform(
      LdMatrixElemShapeMNK{}, typename MMA_Traits<MMA_Op_>::Shape_MNK{}, divides{}));
    using TiledMMA = TiledMMA<MMA_Atom<MMA_Op_>, Layout<Shape<_2, _2, _1>>, Layout<ValShape>>;

    using OperandA_ = OperandA<T, sizeBK, AMajor, ThrCount>;
    using OperandB_ = OperandB<T, sizeBK, BMajor, ThrCount>;
};

template <
  typename TA,
  typename TB,
  typename TC,
  typename Layout_GA,
  typename Layout_GB,
  typename Layout_GC,
  typename Layout_SA,
  typename Layout_SB,
  typename GS_TiledCopy_A,
  typename GS_TiledCopy_B,
  typename SR_TiledCopy_A,
  typename SR_TiledCopy_B,
  typename TiledMMA_>
__global__ void kernel_mma(
  TA const      *A,
  TB const      *B,
  TC            *C,
  Layout_GA      layout_gA,
  Layout_GB      layout_gB,
  Layout_GC      layout_gC,
  Layout_SA      layout_sA,
  Layout_SB      layout_sB,
  GS_TiledCopy_A gs_tiledCopy_A,
  GS_TiledCopy_B gs_tiledCopy_B,
  SR_TiledCopy_A sr_tiledCopy_A,
  SR_TiledCopy_B sr_tiledCopy_B,
  TiledMMA_      tiledMAA)
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

    auto sr_thr_copy_A = sr_tiledCopy_A.get_thread_slice(threadIdx.x);
    auto tCsA          = sr_thr_copy_A.partition_S(sA);
    auto tCrA_view     = sr_thr_copy_A.retile_D(tCrA);

    auto sr_thr_copy_B = sr_tiledCopy_B.get_thread_slice(threadIdx.x);
    auto tCsB          = sr_thr_copy_B.partition_S(sB);
    auto tCrB_view     = sr_thr_copy_B.retile_D(tCrB);

    auto ptr1 = &(tCrA(0));
    copy(sr_tiledCopy_A, tCsA, tCrA_view);
    copy(sr_tiledCopy_B, tCsB, tCrB_view);

    auto fragC = partition_fragment_C(tiledMAA, shape(gC));
    clear(fragC);

    if (thread0()) {
        // clang-format off
        print("gA            : ");print(gA           );print("\n");
        print("gB            : ");print(gB           );print("\n");
        print("gC            : ");print(gC           );print("\n");

        print("sA            : ");print(sA           );print("\n");
        print("sB            : ");print(sB           );print("\n");
        
        print("gs_thr_copy_A : ");print(gs_thr_copy_A);print("\n");
        print("tAgA          : ");print(tAgA         );print("\n");
        print("tAsA          : ");print(tAsA         );print("\n");
        print("gs_thr_copy_B : ");print(gs_thr_copy_B);print("\n");
        print("tBgB          : ");print(tBgB         );print("\n");
        print("tBsB          : ");print(tBsB         );print("\n");
        print("thr_mma       : ");print(thr_mma      );print("\n");
        print("tCrA          : ");print(tCrA         );print("\n");
        print("tCrB          : ");print(tCrB         );print("\n");
        print("tCrC          : ");print(tCrC         );print("\n");
        print("sr_thr_copy_A : ");print(sr_thr_copy_A);print("\n");
        print("tCsA          : ");print(tCsA         );print("\n");
        print("tCrA_view     : ");print(tCrA_view    );print("\n");
        print("sr_thr_copy_B : ");print(sr_thr_copy_B);print("\n");
        print("tCsB          : ");print(tCsB         );print("\n");
        print("tCrB_view     : ");print(tCrB_view    );print("\n");
        print("fragC         : ");print(fragC        );print("\n");
        // clang-format off
    }

    gemm(tiledMAA, fragC, tCrA, tCrB, fragC);
    copy(fragC, tCrC);
}

template <typename GEMM_Config_>
void host_mma() {

    using AConfig         = typename GEMM_Config_::OperandA_;
    using BConfig         = typename GEMM_Config_::OperandB_;
    using ProblemShapeMNK = typename GEMM_Config_::TileShape;

    auto layout_gA = Layout<
      Shape<get_t<0, ProblemShapeMNK>, get_t<2, ProblemShapeMNK>>,
      Stride<get_t<2, ProblemShapeMNK>, _1>>{};
    auto layout_gB = Layout<
      Shape<get_t<1, ProblemShapeMNK>, get_t<2, ProblemShapeMNK>>,
      Stride<get_t<2, ProblemShapeMNK>, _1>>{};
    auto layout_gC = Layout<
      Shape<get_t<0, ProblemShapeMNK>, get_t<1, ProblemShapeMNK>>,
      Stride<get_t<1, ProblemShapeMNK>, _1>>{};

    auto layout_sA = tile_to_shape(typename AConfig::SmemLayout{}, shape(layout_gA));
    auto layout_sB = tile_to_shape(typename BConfig::SmemLayout{}, shape(layout_gB));

    struct SharedStorage {
        cute::array_aligned<tfloat32_t, cute::cosize_v<decltype(layout_sA)>> smem_a;
        cute::array_aligned<tfloat32_t, cute::cosize_v<decltype(layout_sB)>> smem_b;
    };

    static constexpr int SHARED_SIZE = static_cast<int>(sizeof(SharedStorage));

    auto tiled_mma = typename GEMM_Config_::TiledMMA{};

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
      gsTiledCopyA,
      gsTiledCopyB,
      srTiledCopyA,
      srTiledCopyB,
      tiled_mma);
    cudaMemcpy(h_C.data_ptr(), d_C, h_C.numel() * h_C.element_size(), cudaMemcpyDeviceToHost);

    // auto cpu_ans = at::zeros_like(c);
    auto cpu_ans = (h_A.matmul(h_B.mT()));

    // if (test_case == "NN")
    //     cpu_ans = (h_A.matmul(h_B));
    // else if (test_case == "NT")
    //     cpu_ans = (h_A.matmul(h_B.mT()));
    // else if (test_case == "TN")
    //     cpu_ans = (h_A.mT().matmul(h_B));
    // else if (test_case == "TT")
    //     cpu_ans = (h_A.mT().matmul(h_B.mT()));

    std::cout << (h_C.allclose(cpu_ans, 1e-02) ? "MMA Success" : "MMA Failed") << std::endl;
    // std::cout << cpu_ans << std::endl;
    // std::cout << h_C << std::endl;
    // std::cout << (h_C.isclose(cpu_ans, 1e-02)) << std::endl;
}

int main() { host_mma<GemmConfig<tfloat32_t, _16, SM80_16x8x8_F32TF32TF32F32_TN,GenRowMajor,GenColMajor>>(); }
