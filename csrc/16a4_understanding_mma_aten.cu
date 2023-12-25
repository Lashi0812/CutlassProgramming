#include "cute/config.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/stride.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/tfloat.hpp"
#include "cute/underscore.hpp"
#include <ATen/ATen.h>
#include "latex.hpp"

using namespace cute;
using X = Underscore;

constexpr int exponent(int result) {
    int exponent = 0;
    while (result > 1) {
        result = result >> 1; // Right shift the result by 1 bit
        exponent++;
    }
    return exponent;
}

template <class IntT>
Stride<IntT, Int<1>> make_cute_packed_stride(Stride<IntT, Int<1>> s, Shape<int, int> shape_MK) {
    static_assert(
      std::is_integral_v<IntT>,
      "Stride must have an integral type so it can be set dynamically. Static strides not "
      "supported.");
    auto s_copy    = s;
    get<0>(s_copy) = static_cast<IntT>(get<1>(shape_MK));
    return s_copy;
}

template <class IntT>
Stride<Int<1>, IntT> make_cute_packed_stride(Stride<Int<1>, IntT> s, Shape<int, int> shape_MK) {
    static_assert(
      std::is_integral_v<IntT>,
      "Stride must have an integral type so it can be set dynamically. Static strides not "
      "supported.");
    auto s_copy    = s;
    get<1>(s_copy) = static_cast<IntT>(get<0>(shape_MK));
    return s_copy;
}

template <class L>
struct TagToStrideA {
    using type = L;
};

template <>
struct TagToStrideA<GenRowMajor> {
    using type = Stride<int32_t, Int<1>>;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<GenColMajor> {
    using type = Stride<Int<1>, int32_t>;
};

template <class L>
struct TagToStrideB {
    using type = L;
};

template <>
struct TagToStrideB<GenRowMajor> {
    using type = Stride<Int<1>, int32_t>;
};

template <>
struct TagToStrideB<GenColMajor> {
    using type = Stride<int32_t, Int<1>>;
};

template <class LayoutTag>
struct TagToStrideC : TagToStrideA<LayoutTag> {};

// Convenience aliases
template <class LayoutTag>
using TagToStrideA_t = typename TagToStrideA<LayoutTag>::type;

template <class LayoutTag>
using TagToStrideB_t = typename TagToStrideB<LayoutTag>::type;

template <class LayoutTag>
using TagToStrideC_t = typename TagToStrideC<LayoutTag>::type;

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

template <
  typename TAB,
  typename TC_,
  typename sizeBK,
  typename MMA_Op_,
  typename AMajor,
  typename BMajor>
struct GemmConfig {
    using TA                      = TAB;
    using TB                      = TAB;
    using TC                      = TC_;
    using StrideA                 = TagToStrideA_t<AMajor>;
    using StrideB                 = TagToStrideB_t<BMajor>;
    using StrideC                 = TagToStrideC_t<GenRowMajor>;
    using TileShape               = Shape<_128, _128, sizeBK>;
    static int constexpr ThrCount = 128;
    using LdMatrixElemShapeMNK    = Shape<_16, _16, Int<(16 * 2) / sizeof(TA)>>;
    using ValShape                = decltype(transform(
      LdMatrixElemShapeMNK{}, typename MMA_Traits<MMA_Op_>::Shape_MNK{}, divides{}));
    using TiledMMA = TiledMMA<MMA_Atom<MMA_Op_>, Layout<Shape<_2, _2, _1>>, Layout<ValShape>>;

    using OperandA_ = OperandA<TA, sizeBK, AMajor, ThrCount>;
    using OperandB_ = OperandB<TB, sizeBK, BMajor, ThrCount>;
};

template <typename Config>
struct MainLoop {
    using TA             = typename Config::TA;
    using TB             = typename Config::TB;
    using TC             = typename Config::TC;
    using StrideA        = typename Config::StrideA;
    using StrideB        = typename Config::StrideB;
    using StrideC        = typename Config::StrideC;
    using TiledMMA_      = typename Config::TiledMMA;
    using TileShapeMNK   = typename Config::TileShape;
    using Layout_SA      = decltype(tile_to_shape(
      typename Config::OperandA_::SmemLayout{},
      make_shape(get<0>(TileShapeMNK{}), get<2>(TileShapeMNK{}))));
    using Layout_SB      = decltype(tile_to_shape(
      typename Config::OperandB_::SmemLayout{},
      make_shape(get<1>(TileShapeMNK{}), get<2>(TileShapeMNK{}))));
    using GS_TiledCopy_A = typename Config::OperandA_::GSTiledCopy;
    using GS_TiledCopy_B = typename Config::OperandB_::GSTiledCopy;
    using SR_TiledCopy_A =
      decltype(make_tiled_copy_A(typename Config::OperandA_::SRCopyAtom{}, TiledMMA_{}));
    using SR_TiledCopy_B =
      decltype(make_tiled_copy_B(typename Config::OperandB_::SRCopyAtom{}, TiledMMA_{}));

    MainLoop() = default;

    struct Params {
        TA const *ptr_A;
        TB const *ptr_B;
        TC       *ptr_C;
        StrideA   dA;
        StrideB   dB;
        StrideC   dC;
    };
    using Arguments = Params;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const &_, Arguments const &args) {
        return args;
    }

    struct SharedStorage {
        cute::array_aligned<TA, cute::cosize_v<Layout_SA>> smem_a;
        cute::array_aligned<TB, cute::cosize_v<Layout_SB>> smem_b;
    };

    template <typename TensorA, typename TensorB, typename TensorC>
    CUTE_DEVICE void operator()(TensorA gA, TensorB gB, TensorC gC, char *smem) {
        SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem);

        auto           sA = make_tensor(make_smem_ptr(storage.smem_a.data()), Layout_SA{});
        auto           sB = make_tensor(make_smem_ptr(storage.smem_b.data()), Layout_SB{});
        GS_TiledCopy_A gs_tiledCopy_A;
        auto           gs_thr_copy_A = gs_tiledCopy_A.get_thread_slice(threadIdx.x);
        auto           tAgA          = gs_thr_copy_A.partition_S(gA);
        auto           tAsA          = gs_thr_copy_A.partition_D(sA);

        GS_TiledCopy_B gs_tiledCopy_B;
        auto           gs_thr_copy_B = gs_tiledCopy_B.get_thread_slice(threadIdx.x);
        auto           tBgB          = gs_thr_copy_B.partition_S(gB);
        auto           tBsB          = gs_thr_copy_B.partition_D(sB);

        TiledMMA_ tiledMAA;
        auto      thr_mma = tiledMAA.get_thread_slice(threadIdx.x);
        auto      tCrA    = thr_mma.partition_fragment_A(sA);
        auto      tCrB    = thr_mma.partition_fragment_B(sB);
        auto      tCrC    = thr_mma.partition_C(gC);

        SR_TiledCopy_A sr_tiledCopy_A;
        auto           sr_thr_copy_A = sr_tiledCopy_A.get_thread_slice(threadIdx.x);
        auto           tCsA          = sr_thr_copy_A.partition_S(sA);
        auto           tCrA_view     = sr_thr_copy_A.retile_D(tCrA);
        SR_TiledCopy_B sr_tiledCopy_B;
        auto           sr_thr_copy_B = sr_tiledCopy_B.get_thread_slice(threadIdx.x);
        auto           tCsB          = sr_thr_copy_B.partition_S(sB);
        auto           tCrB_view     = sr_thr_copy_B.retile_D(tCrB);

        auto fragC = partition_fragment_C(tiledMAA, take<0, 2>(TileShapeMNK{}));
        clear(fragC);

        for (int k_tile_iter{0}; k_tile_iter < size<2>(gA); ++k_tile_iter) {
            copy(gs_tiledCopy_A, tAgA(_, _, _, 0), tAsA);
            copy(gs_tiledCopy_B, tBgB(_, _, _, 0), tBsB);
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();

            // auto ptr1 = &(tCrA(0));
            copy(sr_tiledCopy_A, tCsA, tCrA_view);
            copy(sr_tiledCopy_B, tCsB, tCrB_view);

            gemm(tiledMAA, fragC, tCrA, tCrB, fragC);
        }

        copy_aligned(fragC, tCrC);
    }
};

template <typename ProblemShapeMNK_, typename MainLoop_>
struct KernelOperator {
    using ProblemShapeMNK = ProblemShapeMNK_;
    using MainLoop        = MainLoop_;
    using MainLoopParams  = typename MainLoop_::Params;
    using TileShapeMNK    = typename MainLoop_::TileShapeMNK;

    KernelOperator() = default;
    struct Params {
        ProblemShapeMNK problem_shape;
        MainLoopParams  mainLoop;
    };

    using Arguments = Params;

    static Params to_underlying_arguments(Arguments const &args) {
        return {
          args.problem_shape, MainLoop::to_underlying_arguments(args.problem_shape, args.mainLoop)};
    }

    CUTE_DEVICE void operator()(Params params, char *smem) {
        auto mA = make_tensor(
          make_gmem_ptr(params.mainLoop.ptr_A),
          make_shape(get<0>(params.problem_shape), get<2>(params.problem_shape)),
          params.mainLoop.dA);
        auto mB = make_tensor(
          make_gmem_ptr(params.mainLoop.ptr_B),
          make_shape(get<1>(params.problem_shape), get<2>(params.problem_shape)),
          params.mainLoop.dB);
        auto mC = make_tensor(
          make_gmem_ptr(params.mainLoop.ptr_C),
          make_shape(get<0>(params.problem_shape), get<1>(params.problem_shape)),
          params.mainLoop.dC);

        auto blk_shape                   = TileShapeMNK{};
        auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
        auto blk_coord                   = make_coord(m_coord, n_coord, _);

        auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X, _1>{});
        auto gB = local_tile(mB, blk_shape, blk_coord, Step<X, _1, _1>{});
        auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1, _1, X>{});

        MainLoop mainLoop;
        mainLoop(gA, gB, gC, smem);
    }
};

template <typename Operator>
__global__ void kernel_mma(typename Operator::Params params) {
    extern __shared__ char smem[];
    Operator               op;
    op(params, smem);
}

template <typename Kernel_>
struct KernelAdapter {
    using Operator        = Kernel_;
    using Loop            = typename Operator::MainLoop;
    using Params_         = typename Operator::Params;
    using ProblemShapeMNK = typename Operator::ProblemShapeMNK;
    using TA              = typename Loop::TA;
    using TB              = typename Loop::TB;
    using TC              = typename Loop::TC;
    using StrideA         = typename Loop::StrideA;
    using StrideB         = typename Loop::StrideB;
    using StrideC         = typename Loop::StrideC;

    StrideA strideA;
    StrideB strideB;
    StrideC strideC;

    Params_ params_;

    void initialize(ProblemShapeMNK problem_size, TA const *d_A, TB const *d_B, TC *d_C) {

        auto M = cute::size<0>(problem_size);
        auto N = cute::size<1>(problem_size);
        auto K = cute::size<2>(problem_size);

        strideA = make_cute_packed_stride(StrideA{}, make_shape(M, K));
        strideB = make_cute_packed_stride(StrideB{}, make_shape(N, K));
        strideC = make_cute_packed_stride(StrideC{}, make_shape(M, N));

        auto args =
          typename Operator::Arguments{problem_size, {d_A, d_B, d_C, strideA, strideB, strideC}};

        params_ = Operator::to_underlying_arguments(args);
    }

    void run(ProblemShapeMNK problem_size, TA const *d_A, TB const *d_B, TC *d_C) {
        initialize(problem_size, d_A, d_B, d_C);
        static int constexpr SHARED_SIZE = sizeof(typename Loop::SharedStorage{});
        kernel_mma<Operator><<<1, 128, SHARED_SIZE>>>(params_);
    }
};

template <typename GEMM_Config_>
void host_mma() {

    auto M = 128;
    auto N = 128;
    auto K = 16;

    auto problemShapeMNK = make_shape(M, N, K);

    auto h_A = at::rand({M, K}, at::TensorOptions().dtype(at::kFloat));

    auto h_B = at::rand({N, K}, at::TensorOptions().dtype(at::kFloat));

    auto h_C = at::zeros({M, N}, at::TensorOptions().dtype(at::kFloat));

    tfloat32_t *d_A, *d_B;
    float      *d_C;
    cudaMalloc((void **)&d_A, h_A.numel() * h_A.element_size());
    cudaMalloc((void **)&d_B, h_B.numel() * h_B.element_size());
    cudaMalloc((void **)&d_C, h_C.numel() * h_C.element_size());

    cudaMemcpy(d_A, h_A.data_ptr(), h_A.numel() * h_A.element_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data_ptr(), h_B.numel() * h_B.element_size(), cudaMemcpyHostToDevice);

    using Loop     = MainLoop<GEMM_Config_>;
    using Operator = KernelOperator<Shape<int, int, int>, Loop>;
    using Adapter  = KernelAdapter<Operator>;

    Adapter ap;
    ap.run(problemShapeMNK, d_A, d_B, d_C);
    cudaDeviceSynchronize();

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
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // host_mma<GemmConfig<
    //   tfloat32_t,
    //   float,
    //   _16,
    //   SM80_16x8x8_F32TF32TF32F32_TN,
    //   GenRowMajor,
    //   GenColMajor>>();

    using test = OperandA<tfloat32_t, _16, GenRowMajor, 128>;
    std::cout << "%% " << test::ThrMode0 << " , " << test::ThrMode1 << std::endl;
    print_latex(typename test::GSTiledCopy{}, "GSTiledCopy_T8x16_V1x4",3);
}
