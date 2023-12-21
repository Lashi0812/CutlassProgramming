#include "cute/stride.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/numeric/int.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/tfloat.hpp"
#include "latex.hpp"
#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>
#include <numeric>
#include <string>
#include <vector>

using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                          Understanding SR tiledCpy and tiled MMA
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MMA_Atom_OP_, typename Atom_Layout, typename Val_Layout, typename Copy_Atom_Op_>
void test_sr_tiled_copy_and_tiled_mma(std::string test_name, int ps = -1) {
    auto tiled_mma = TiledMMA<MMA_Atom<MMA_Atom_OP_>, Atom_Layout, Val_Layout>{};

    // A
    auto ivecA = std::vector<int>(128 * 32);
    std::iota(ivecA.begin(), ivecA.end(), 0);
    auto smem_layoutA = tile_to_shape(
      composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}),
      make_shape(_128{}, _32{}));
    auto smem_tensorA   = make_tensor(ivecA.data(), smem_layoutA);
    auto sr_tiled_copyA = make_tiled_copy_A(Copy_Atom<Copy_Atom_Op_, tfloat32_t>{}, tiled_mma);

    // B
    auto ivecB = std::vector<int>(128 * 32);
    std::iota(ivecB.begin(), ivecB.end(), 0);
    auto smem_layoutB = tile_to_shape(
      composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}),
      make_shape(_128{}, _32{}));
    auto smem_tensorB   = make_tensor(ivecB.data(), smem_layoutB);
    auto sr_tiled_copyB = make_tiled_copy_B(Copy_Atom<Copy_Atom_Op_, tfloat32_t>{}, tiled_mma);

    if (ps == -1) {
        auto thr_id = std::vector<int>{0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127};
        // A
        for (int i = 0; i < thr_id.size(); ++i) {
            auto thr_copy = sr_tiled_copyA.get_thread_slice(thr_id[i]);
            auto partA    = thr_copy.partition_S(smem_tensorA);
            print("%% Copy Part %d  : ", thr_id[i]);
            print_tensor(partA);
            print("\n");
        }

        for (int i = 0; i < thr_id.size(); ++i) {
            auto thr_mma = tiled_mma.get_thread_slice(thr_id[i]);
            auto partA   = thr_mma.partition_A(smem_tensorA);
            print("%% MMA Part %d  : ", thr_id[i]);
            print_tensor(partA);
            print("\n");
        }

        // B
        print("%% ******************** B starts **********************");
        for (int i = 0; i < thr_id.size(); ++i) {
            auto thr_copy = sr_tiled_copyB.get_thread_slice(thr_id[i]);
            auto partB    = thr_copy.partition_S(smem_tensorA);
            print("%% Copy Part %d  : ", thr_id[i]);
            print_tensor(partB);
            print("\n");
        }

        for (int i = 0; i < thr_id.size(); ++i) {
            auto thr_mma = tiled_mma.get_thread_slice(thr_id[i]);
            auto partB   = thr_mma.partition_B(smem_tensorA);
            print("%% MMA Part %d  : ", thr_id[i]);
            print_tensor(partB);
            print("\n");
        }
    }

    if (ps == 1) {
        // clang-format off
        // A
        print("%% SMEM_LAYOUT_A : ");print_latex(       smem_layoutA                     ,("tiled_copy_and_tiled_mma_"+test_name+"_smem_layoutA" ).c_str());print("\n");
        print("%% LAYOUT_A_MK : "  );print_latex(get<0>(tiled_mma      .get_layoutA_MK()),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_A_MK"  ).c_str());print("\n");
        print("%% LAYOUT_A_TV : "  );print_latex(       tiled_mma      .get_layoutA_TV( ),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_A_TV"  ).c_str());print("\n");
        print("%% LAYOUT_A_S_TV : ");print_latex(       sr_tiled_copyA .get_layoutS_TV( ),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_A_S_TV").c_str());print("\n");
        print("%% LAYOUT_A_S_MN : ");print_latex(get<0>(sr_tiled_copyA .get_layoutS_MN()),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_A_S_MN").c_str());print("\n");
        print("%% LAYOUT_A_D_TV : ");print_latex(       sr_tiled_copyA .get_layoutD_TV( ),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_A_D_TV").c_str());print("\n");
        print("%% LAYOUT_A_D_MN : ");print_latex(get<0>(sr_tiled_copyA .get_layoutD_MN()),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_A_D_MN").c_str());print("\n");
        // B
        print("%% SMEM_LAYOUT_B : ");print_latex(       smem_layoutB                     ,("tiled_copy_and_tiled_mma_"+test_name+"_smem_layoutB" ).c_str());print("\n");
        print("%% LAYOUT_B_MK : "  );print_latex(get<0>(tiled_mma      .get_layoutB_NK()),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_B_MK"  ).c_str());print("\n");
        print("%% LAYOUT_B_TV : "  );print_latex(       tiled_mma      .get_layoutB_TV( ),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_B_TV"  ).c_str());print("\n");
        print("%% LAYOUT_B_S_TV : ");print_latex(       sr_tiled_copyB .get_layoutS_TV( ),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_B_S_TV").c_str());print("\n");
        print("%% LAYOUT_B_S_MN : ");print_latex(get<0>(sr_tiled_copyB .get_layoutS_MN()),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_B_S_MN").c_str());print("\n");
        print("%% LAYOUT_B_D_TV : ");print_latex(       sr_tiled_copyB .get_layoutD_TV( ),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_B_D_TV").c_str());print("\n");
        print("%% LAYOUT_B_D_MN : ");print_latex(get<0>(sr_tiled_copyB .get_layoutD_MN()),("tiled_copy_and_tiled_mma_"+test_name+"_LAYOUT_B_D_MN").c_str());print("\n");
        // clang-format on
    }
}

void test_sr_tiled_copy_and_tiled_mma_examples(int ps = -1) {
    if (ps == 1)
        print_latex_header();
    test_sr_tiled_copy_and_tiled_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_2, _2, _1>>,
      Layout<Shape<_1, _2, _1>>,
      SM75_U32x4_LDSM_N>("A2x2x1_V1x2x1", ps);
    if (ps == 1)
        print_latex_footer();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                          Understanding the TV layout in Tiled Copy

// * TVLayout --> (thr,val)
// * thr shape -->  this will be in the right inverse order
// * (_32,_4):(_4,_1)
// * ie. find the index of 1 (1)in the stride
//              --> multiply the stride value (1) and shape value (4)at found index = A (4)
// * Now find the index of A (0) in the stride
//              --> multiply the stride value (4) and shape value (32) at found index = B (128)
// * repeat
// ? thr shape will be value at 1,0 in shape --> (4,32)

// * val shape --> same as the val layout shape
// * TVlayout stride  -->
//      Start from val stride and mul val and stride put in thr stride

// ThrLayout : (_32,_4):(_1,_32)
// ValLayout : (_1,_4):(_0,_1)
// TVLayout  : ((_32,_4),_4):((_1,_128),_32)
//
// ThrLayout : (_32,_4):(_1,_32)
// ValLayout : (_2,_4):(_1,_2)
// TVLayout  : ((_32,_4),(_2,_4)):((_2,_256),(_1,_64))
//
// ThrLayout : (_32,_4):(_4,_1)
// ValLayout : (_1,_4):(_0,_1)
// TVLayout  : ((_4,_32),_4):((_128,_1),_32)
//
// ThrLayout : (_32,_4):(_4,_1)
// ValLayout : (_2,_4):(_1,_2)
// TVLayout  : ((_4,_32),(_2,_4)):((_256,_2),(_1,_64))
//
// ThrLayout : (_32,_4):(_4,_1)
// ValLayout : (_1,_8):(_0,_1)
// TVLayout  : ((_4,_32),_8):((_256,_1),_32)
//
// ThrLayout : (_32,_4):(_4,_1)
// ValLayout : (_2,_8):(_1,_2)
// TVLayout  : ((_4,_32),(_2,_8)):((_512,_2),(_1,_64))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ThrLayout, typename ValLayout>
void test_layout_tv() {
    using tiled_copy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, tfloat32_t>{}, ThrLayout{}, ValLayout{}));

    // clang-format off
    print("ThrLayout : ");print(ThrLayout{});print("\n");
    print("ValLayout : ");print(ValLayout{});print("\n");
    print("TVLayout  : ");print_layout(typename tiled_copy::TiledLayout_TV{});print("\n");
    print("******************************************\n");
    // clang-format on
}

void test_layout_tv_examples() {
    test_layout_tv<Layout<Shape<_2, _4>, Stride<_1, _2>>, Layout<Shape<_1, _4>>>();
    test_layout_tv<Layout<Shape<_2, _4>, Stride<_1, _2>>, Layout<Shape<_2, _4>>>();
    test_layout_tv<Layout<Shape<_2, _4>, Stride<_4, _1>>, Layout<Shape<_1, _4>>>();
    test_layout_tv<Layout<Shape<_2, _4>, Stride<_4, _1>>, Layout<Shape<_2, _4>>>();
    test_layout_tv<Layout<Shape<_2, _4>, Stride<_4, _1>>, Layout<Shape<_1, _8>>>();
    test_layout_tv<Layout<Shape<_2, _4>, Stride<_4, _1>>, Layout<Shape<_2, _8>>>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                  Understanding the Source and destination partition of tiled copy
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ThrLayout, typename ValLayout, typename SrcLayout, typename DstLayout>
void test_source_and_dest_partition() {
    using TiledCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, tfloat32_t>{}, ThrLayout{}, ValLayout{}));

    auto src_tensor = make_counting_tensor(SrcLayout{});
    auto dst_tensor = make_counting_tensor(DstLayout{});

    TiledCopy tiled_copy;
    auto      thr_copy = tiled_copy.get_thread_slice(0);
    auto      tSgS     = thr_copy.partition_S(src_tensor);
    auto      tDgD     = thr_copy.partition_D(dst_tensor);

    // clang-format off
    print("ThrLayout     : ");print       (ThrLayout{}                         );print("\n");
    print("ValLayout     : ");print       (ValLayout{}                         );print("\n");
    print("Tiler_MN      : ");print       (typename  TiledCopy::Tiler_MN{}     );print("\n");
    print("AtomLayoutSrc : ");print       (typename  TiledCopy::AtomLayoutSrc{});print("\n");
    print("AtomLayoutDst : ");print       (typename  TiledCopy::AtomLayoutDst{});print("\n");
    print("SrcLayout     : ");print       (SrcLayout{}                         );print("\n");
    print("DstLayout     : ");print       (DstLayout{}                         );print("\n");
    print("tSgS          : ");print_tensor(tSgS                                );print("\n");
    print("tDgD          : ");print_tensor(tDgD                                );print("\n");
    print("******************************************\n");
    // clang-format on
}

void test_source_and_dest_partition_examples() {
    test_source_and_dest_partition<
      Layout<Shape<_2, _4>, Stride<_1, _2>>,
      Layout<Shape<_2, _1>>,
      Layout<Shape<_16, _16>, Stride<_1, _16>>,
      Layout<Shape<_8, _2>, Stride<_1, _8>>>();
    test_source_and_dest_partition<
      Layout<Shape<_2, _4>, Stride<_1, _2>>,
      Layout<Shape<_2, _2>>,
      Layout<Shape<_16, _16>, Stride<_1, _16>>,
      Layout<Shape<_8, _2>, Stride<_1, _8>>>();
    test_source_and_dest_partition<
      Layout<Shape<_2, _4>, Stride<_4, _1>>,
      Layout<Shape<_1, _2>>,
      Layout<Shape<_16, _16>, Stride<_16, _1>>,
      Layout<Shape<_8, _2>, Stride<_2, _1>>>();
    test_source_and_dest_partition<
      Layout<Shape<_2, _4>, Stride<_4, _1>>,
      Layout<Shape<_2, _2>>,
      Layout<Shape<_16, _16>, Stride<_16, _1>>,
      Layout<Shape<_8, _2>, Stride<_2, _1>>>();
    test_source_and_dest_partition<
      Layout<Shape<_2, _4>, Stride<_4, _1>>,
      Layout<Shape<_1, _4>>,
      Layout<Shape<_16, _16>, Stride<_16, _1>>,
      Layout<Shape<_8, _2>, Stride<_2, _1>>>();
    test_source_and_dest_partition<
      Layout<Shape<_2, _4>, Stride<_4, _1>>,
      Layout<Shape<_2, _4>>,
      Layout<Shape<_16, _16>, Stride<_16, _1>>,
      Layout<Shape<_8, _2>, Stride<_2, _1>>>();
}

int main(int argc, char *argv[]) {
    // print_select
    [[maybe_unused]] int ps{-1};
    if (argc >= 2)
        ps = atoi(argv[1]);
    // test_sr_tiled_copy_and_tiled_mma_examples(ps);
    // test_layout_tv_examples();
    test_source_and_dest_partition_examples();
}