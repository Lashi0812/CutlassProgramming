#include <cute/tensor.hpp>
#include "cute/arch/mma_sm80.hpp"
#include "cute/underscore.hpp"
#include "cute/layout.hpp"
#include "cute/atom/copy_atom.hpp"
#include <cute/atom/mma_atom.hpp>
#include <latex.hpp>

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////
//                      Tiled MMA
////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename AtomLayoutMNK,
  typename ValLayoutMNK,
  typename PermutationMNK = Tile<Underscore, Underscore, Underscore>,
  typename MMA_Op_        = SM80_16x8x16_F32F16F16F32_TN>
void test_tiled_mma(const char *test_name, int ps = -1) {
    using tiled_mma = TiledMMA<MMA_Atom<MMA_Op_>, AtomLayoutMNK, ValLayoutMNK, PermutationMNK>;

    // clang-format off
    print("%% AtomLayoutMNK   : "  );print(AtomLayoutMNK                           {});print("\n");
    print("%% ValLayoutMNK    : "  );print(ValLayoutMNK                            {});print("\n");
    print("%% AtomShape_MNK   :  " );print(typename      tiled_mma::AtomShape_MNK  {});print("\n");
    print("%% AtomLayoutC_TV  :  " );print(typename      tiled_mma::AtomLayoutC_TV {});print("\n");
    print("%% AtomLayoutA_TV  :  " );print(typename      tiled_mma::AtomLayoutA_TV {});print("\n");
    print("%% AtomLayoutB_TV  :  " );print(typename      tiled_mma::AtomLayoutB_TV {});print("\n");
    print("%% AtomThrID       :  " );print(typename      tiled_mma::AtomThrID      {});print("\n");
    print("%% TiledShape_MNK  :  " );print(typename      tiled_mma::TiledShape_MNK {});print("\n");
    print("%% ThrLayoutVMNK   :  " );print(typename      tiled_mma::ThrLayoutVMNK  {});print("\n");
    print("%% TidLayout       :  " );print(typename      tiled_mma::TidLayout      {});print("\n");
    // clang-format on
    if (ps == 1)
        print_latex(tiled_mma{}, test_name);
}

void test_tiled_mma_examples(int ps = -1) {
    if (ps == 1)
        print_latex_header();
    // {
    //     test_tiled_mma<
    //       Layout<Shape<_1, _1, _1>>,
    //       Layout<Shape<_1, _1, _1>>,
    //       Tile<Underscore, Underscore, Underscore>>("mma_01_A111_V111_PXXX", ps);
    //     print("\n\n");
    // }
    // {
    //     test_tiled_mma<Layout<Shape<_2, _1, _1>>, Layout<Shape<_1, _1, _1>>>(
    //       "mma_02_A211_V111_PXXX", ps);
    //     print("\n\n");
    // }
    // {
    //     test_tiled_mma<Layout<Shape<_1, _1, _1>>, Layout<Shape<_2, _1, _1>>>(
    //       "mma_03_A111_V211_PXXX", ps);
    //     print("\n\n");
    // }
    // {
    //     test_tiled_mma<Layout<Shape<_2, _2, _1>>, Layout<Shape<_1, _1, _1>>>(
    //       "mma_04_A221_V111,PXXX", ps);
    //     print("\n\n");
    // }
    // {
    //     test_tiled_mma<Layout<Shape<_2, _2, _1>>, Layout<Shape<_2, _2, _1>>>(
    //       "mma_05_A221_V221_PXXX", ps);
    //     print("\n\n");
    // }

    // {
    //     test_tiled_mma<
    //       Layout<Shape<_1, _1, _1>>,
    //       Layout<Shape<_1, _1, _1>>,
    //       Tile<Layout<_2, _2>, Underscore, Underscore>>("mma_06_A111_V111_P22XX", ps);
    //     print("\n\n");
    // }
    // {
    //     test_tiled_mma<
    //       Layout<Shape<_2, _1, _1>>,
    //       Layout<Shape<_1, _1, _1>>,
    //       Tile<Underscore, Layout<_2, _4>, Underscore>>("mma_07_A211_V111_PX24X", ps);
    //     print("\n\n");
    // }
    // {
    //     test_tiled_mma<
    //       Layout<Shape<_1, _1, _1>>,
    //       Layout<Shape<_2, _1, _1>>,
    //       Tile<Underscore, Underscore, Layout<_2, _8>>>("mma_08_A111_V211_PXX28", ps);
    //     print("\n\n");
    // }

    {
        test_tiled_mma<
          Layout<Shape<_2, _2, _1>>,
          Layout<Shape<_1, _2, _1>>,
          Tile<Underscore, Underscore, Underscore>,
          SM80_16x8x8_F32TF32TF32F32_TN>("mma_09_A221_V121_PXXX", ps);
        print("\n\n");
    }
    if (ps == 1)
        print_latex_footer();
}

////////////////////////////////////////////////////////////////////////////////////////////
//                      Thr MMA
////////////////////////////////////////////////////////////////////////////////////////////

void test_thr_mma(int tid_idx) {
    auto tiled_mma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>{};
    auto thr_mma   = tiled_mma.get_slice(tid_idx);

    auto tensorC = make_counting_tensor(get<0>(tiled_mma.get_layoutC_MN()));
    print(tensorC);
    auto part_c = thr_mma.partition_C(tensorC);
    print_tensor(part_c);
}

void test_thr_mma_examples() {
    test_thr_mma(0);
    test_thr_mma(1);
}

int main(int argc, char *argv[]) {
    [[maybe_unused]] int ps{-1};
    if (argc >= 2)
        ps = atoi(argv[1]);
    test_tiled_mma_examples(ps);
    // test_thr_mma_examples();
}