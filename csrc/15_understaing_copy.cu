#include "cute/tensor.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cute/util/print.hpp"
#include "latex.hpp"
#include <cstdint>
#include <cute/algorithm/copy.hpp>

using namespace cute;

template <typename Args>
void custom_print(Args args, int ps = -1) {
    switch (ps) {
        case 0:
            print_layout(args);
            break;
        case 1:
            print_latex(args);
            break;
        default:
            print(args);
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                             Start of Copy traits
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename S, typename D>
void test_copy_trait() {
    using copy_trait = Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<S, D>>;

    print("Thread layout : ");
    print(typename copy_trait::ThrID{});
    print("\n");

    print("Src layout : ");
    print_layout(typename copy_trait::SrcLayout{});
    print("\n");

    print("Dst layout : ");
    print_layout(typename copy_trait::DstLayout{});
    print("\n");

    print("Ref layout : ");
    print_layout(typename copy_trait::RefLayout{});
    print("\n");
}

void test_copy_trait_examples() {
    {
        print("Int2 Bit Representation : \n");
        test_copy_trait<int2_t, int2_t>();
    }
    {
        print("Int8 Bit Representation : \n");
        test_copy_trait<int4_t, int4_t>();
    }
    {
        print("Int8 Bit Representation : \n");
        test_copy_trait<int8_t, int8_t>();
    }
    {
        print("Int16 Bit Representation : \n");
        test_copy_trait<int16_t, int16_t>();
    }
    {
        print("Int32 Bit Representation : \n");
        test_copy_trait<int, int>();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              End of Copy traits
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Start of Copy Atom
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename S, typename D, typename T>
void test_copy_atom() {
    using copy_atom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<S, D>>, T>;

    print("Bit Source Layout : \n");
    print_latex(typename copy_atom::BitLayoutSrc{});
    print("\n");
    print("Value Source Layout : \n");
    print_latex(typename copy_atom::ValLayoutSrc{});
    print("\n");

    print("Bit Destination Layout : \n");
    print_latex(typename copy_atom::BitLayoutDst{});
    print("\n");
    print("Value Destination Layout : \n");
    print_latex(typename copy_atom::ValLayoutDst{});
    print("\n");

    print("Bit Reference Layout : \n");
    print_latex(typename copy_atom::BitLayoutRef{});
    print("\n");
    print("Value Reference Layout : \n");
    print_latex(typename copy_atom::ValLayoutRef{});
    print("\n");
}

void test_copy_atom_examples() {
    { test_copy_atom<int, int, int8_t>(); }
    { test_copy_atom<uint128_t, uint128_t, half_t>(); }
    { test_copy_atom<uint128_t, uint128_t, int4_t>(); }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//                              End of Copy Atom
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Make tiled Copy
//  Produce the TiledCopy from the logical thread and Value layouts
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ThrLayout,
  typename ValLayout,
  typename Packed = uint128_t,
  typename Original = half_t>
void test_make_tiled_copy(char const *test_name = {}, int ps = 0) {
    auto res = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<Packed>, Original>{}, ThrLayout{}, ValLayout{});

    if (ps == 1) {
        print_latex(res, test_name);
    } else {
        print("Thread Layout : ");
        print(ThrLayout{});
        print("\n");
        print("Value  Layout : ");
        print(ValLayout{});
        print("\n");
        print(res);
    }
}

void test_make_tiled_copy_examples(int ps = 0) {
    if (ps == 1)
        print_latex_header();
    test_make_tiled_copy<
      Layout<Shape<_16, _8>, Stride<_8, _1>>,
      Layout<Shape<_1, _8>, Stride<_0, _1>>>("CP_T16x8_V1x8", ps);

    test_make_tiled_copy<Layout<Shape<_4, _8>>, Layout<Shape<_4, _2>>, uint64_t, half_t>(
      "CP_T4x8_V4x2", ps);

    test_make_tiled_copy<Layout<Shape<_8, _4>>, Layout<Shape<_2, _4>>, uint32_t, half_t>(
      "CP_T8x4_V2x4_Pa32B_Or16B", ps);
    if (ps == 1)
        print_latex_footer();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Understanding the function in Tiled Copy
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ThrLayout,
  typename ValLayout,
  typename Packed = uint128_t,
  typename Original = half_t>
void test_get_layouts() {
    auto tiled_copy = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<Packed>, Original>{}, ThrLayout{}, ValLayout{});

    // clang-format off
    print("Thread Layout : ");print(ThrLayout{});print("\n");
    print("Value  Layout : ");print(ValLayout{});print("\n");
    print(tiled_copy);
    print("Layouts S_TV : ");print(tiled_copy.get_layoutS_TV());print("\n");
    print("Layouts S_MN : ");print(tiled_copy.get_layoutS_MN());print("\n");
    print("Layouts D_TV : ");print(tiled_copy.get_layoutD_TV());print("\n");
    print("Layouts D_MN : ");print(tiled_copy.get_layoutD_MN());print("\n");
    // clang-format on
}

void test_get_layouts_examples() {
    test_get_layouts<
      Layout<Shape<_16, _8>, Stride<_8, _1>>,
      Layout<Shape<_1, _8>, Stride<_0, _1>>>();
}

template <typename ThrLayout, typename ValLayout, typename SrcLayout>
void test_tile2frag(int ps) {
    auto tiled_copy = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{}, ThrLayout{}, ValLayout{});

    auto thrFrag = tiled_copy.tidfrg_S(SrcLayout{});

    // clang-format off
    print("Thread Layout : ");custom_print(ThrLayout{},ps);print("\n");
    print("Value  Layout : ");custom_print(ValLayout{},ps);print("\n");
    print(tiled_copy);
    print("Source Layout : ");custom_print(SrcLayout{},ps);print("\n");
    print("Thread Frag   : ");print(thrFrag    );print("\n");
    // clang-format on
}

void test_tile2frag_examples(int ps) {
    test_tile2frag<
      Layout<Shape<_16, _8>, Stride<_8, _1>>,
      Layout<Shape<_1, _4>, Stride<_0, _1>>,
      Layout<Shape<_16, _64>>>(ps);
}

template <typename ThrLayout, typename ValLayout, typename SrcLayout, int tidx>
void test_get_slice(int ps) {
    auto tiled_copy = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>{}, ThrLayout{}, ValLayout{});

    auto tid2Frag = tiled_copy.tidfrg_S(SrcLayout{});

    auto src_tensor = make_counting_tensor(SrcLayout{});
    auto thr_slice = tiled_copy.get_slice(tidx);
    auto part_s = thr_slice.partition_S(src_tensor);

    // clang-format off
    print("Thread Layout : ");custom_print(ThrLayout{},ps);print("\n");
    print("Value  Layout : ");custom_print(ValLayout{},ps);print("\n");
    print(tiled_copy);
    print("Source Layout : ");custom_print(SrcLayout{},ps);print("\n");
    print("Thread Frag   : ");print(tid2Frag    );print("\n");
    print(tid2Frag(tidx,_, (_,_)));
    print("Partition     : ");print_tensor(part_s    );print("\n");
    // clang-format on
}

void test_get_slice_examples(int ps) {
    test_get_slice<
      Layout<Shape<_16, _8>, Stride<_8, _1>>,
      Layout<Shape<_1, _4>, Stride<_0, _1>>,
      Layout<Shape<_16, _64>>,
      0>(ps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                      LD Matrix Traits
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Operation>
void test_ldmatrix_traits(char const *test_name) {
    using traits = Copy_Traits<Operation>;

    print_latex(typename traits::SrcLayout{}, test_name);
    print_latex(typename traits::DstLayout{}, test_name);
    print_latex(typename traits::RefLayout{}, test_name);
}

void test_ldmatrix_traits_examples() {
    print_latex_header();
    test_ldmatrix_traits<SM75_U32x1_LDSM_N>("SM75_U32x1_LDSM_N");
    test_ldmatrix_traits<SM75_U32x2_LDSM_N>("SM75_U32x2_LDSM_N");
    test_ldmatrix_traits<SM75_U32x4_LDSM_N>("SM75_U32x4_LDSM_N");
    test_ldmatrix_traits<SM75_U16x2_LDSM_T>("SM75_U16x2_LDSM_T");
    test_ldmatrix_traits<SM75_U16x4_LDSM_T>("SM75_U16x4_LDSM_T");
    test_ldmatrix_traits<SM75_U16x8_LDSM_T>("SM75_U16x8_LDSM_T");
    print_latex_footer();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  LD Matrix copy Atom
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operation, typename T, typename ThrLayout, typename ValLayout>
void test_ldmatrix_copy_atom(char const *test_name) {
    using atom = decltype(make_tiled_copy(Copy_Atom<Operation, T>{}, ThrLayout{}, ValLayout{}));

    print_latex(atom{}, test_name);
}

void test_ldmatrix_copy_atom_examples() {
    print_latex_header();
    test_ldmatrix_copy_atom<
      SM75_U32x1_LDSM_N,
      half_t,
      Layout<Shape<_4, _8>>,
      Layout<Shape<_2, _1>>>("ldmatrix_num1_T4x8_V2x1");

    test_ldmatrix_copy_atom<
      SM75_U32x1_LDSM_N,
      half_t,
      Layout<Shape<_8, _4>>,
      Layout<Shape<_1, _2>>>("ldmatrix_num1_T8x4_V1x2");

    test_ldmatrix_copy_atom<
      SM75_U32x1_LDSM_N,
      half_t,
      Layout<Shape<_32, _1>>,
      Layout<Shape<_1, _8>>>("ldmatrix_num1_T32x1_V1x8");

    test_ldmatrix_copy_atom<
      SM75_U32x2_LDSM_N,
      half_t,
      Layout<Shape<_32, _1>>,
      Layout<Shape<_1, _4>>>("ldmatrix_num2_T32x1_V1x4");
    print_latex_footer();
}

int main(int argc, char *argv[]) {
    // print_select
    [[maybe_unused]] int ps{-1};
    if (argc >= 2)
        ps = atoi(argv[1]);

    // test_copy_trait_examples();
    // test_copy_atom_examples();
    // test_make_tiled_copy_examples(ps);
    // test_get_layouts_examples();
    // test_tile2frag_examples(ps);
    // test_get_slice_examples(ps);
    // test_ldmatrix_traits_examples();
    test_ldmatrix_copy_atom_examples();
}