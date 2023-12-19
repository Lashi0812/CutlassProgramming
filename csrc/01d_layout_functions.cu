#include "cute/tensor.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/container/tuple.hpp"
#include "cute/int_tuple.hpp"
#include "cute/numeric/int.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/util/print.hpp"
#include "cutlass/half.h"
#include "latex.hpp"
#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>
#include <numeric>
#include <string>
#include <vector>

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
//                              Transform Leaf
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tuple, typename Fn>
void test_transform_leaf() {
    auto res = transform_leaf(Tuple{}, Fn{});

    // clang-format off
    print("Input  : ");print(Tuple{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_transform_leaf_examples() {
    test_transform_leaf<tuple<_2, _m1, tuple<_m5, _1>>, abs_fn>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Transform
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TupleA, typename TupleB, typename Fn>
void test_transform() {
    auto res = transform(TupleA{}, TupleB{}, Fn{});

    // clang-format off
    print("Input  : ");print(TupleA{});print(" , ");print(TupleB{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_transform_examples() {
    {
        print("Addition : \n");
        test_transform<tuple<_2, _10>, tuple<_5, _6>, plus>();
    }
    {
        print("Max  : \n");
        test_transform<tuple<_2, _10>, tuple<_5, _6>, max_fn>();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Find
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename X>
void test_find() {
    auto res = find(T{}, X{});

    // clang-format off
    print("Input  : ");print(T{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_find_examples() {
    test_find<tuple<_1, _4, _5, _2, _6>, _4>();
    test_find<tuple<_1, _4, _5, _2, _6>, _5>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Find if
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename F>
void test_find_if(T t, F &&f) {
    auto res = find_if(T{}, f);

    // clang-format off
    print("Input  : ");print(T{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_find_if_examples() {
    {
        print("Find the Value which greater than 5 and return index : \n");
        test_find_if(tuple<_1, _2, _5, _7>{}, [&](auto const &i) { return greater{}(i, _5{}); });
    }
    {
        print("Find the Value which equal to 5 and return index : \n");
        test_find_if(tuple<_1, _2, _5, _7>{}, [&](auto const &i) { return equal_to{}(i, _5{}); });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Compact Col Major
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Shape>
void test_compact_col_major() {
    auto res = compact_col_major(Shape{});
    print("Input  : ");
    print(Shape{});
    print("\n");
    print("Output : ");
    print(res);
    print("\n");
}

void test_compact_col_major_examples() {
    test_compact_col_major<Shape<_4>>();
    test_compact_col_major<Shape<_4, _2>>();
    test_compact_col_major<Shape<_4, _2, _8>>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Inverse seq
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Start, typename Shape, typename Stride>
void test_inverse_seq() {
    auto res = detail::inverse_seq<Start>(Shape{}, Stride{}, seq<>{});

    // clang-format off
    print("Input  : ");print(Shape{});print(Stride{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_inverse_seq_examples() {
    test_inverse_seq<1, tuple<_4>, tuple<_1>>();
    test_inverse_seq<1, tuple<_4, _4>, tuple<_1, _4>>();
    test_inverse_seq<1, tuple<_4, _4>, tuple<_1, _5>>();
    test_inverse_seq<1, tuple<_4, _5>, tuple<_1, _4>>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Right Inverse
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
void test_right_inverse(int ps) {
    auto res = right_inverse(Layout{});

    // clang-format off
    print("Input  : ");print(Layout{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_right_inverse_examples(int ps) {
    // test_right_inverse<Layout<Shape<_4>, Stride<_1>>>(ps);
    // test_right_inverse<Layout<Shape<_4, _4>, Stride<_1, _4>>>(ps);
    // test_right_inverse<Layout<Shape<_4, _4>, Stride<_1, _5>>>(ps);
    // test_right_inverse<Layout<Shape<_4, _5>, Stride<_1, _4>>>(ps);
    // test_right_inverse<Layout<Shape<_4, _5>, Stride<_5, _1>>>(ps);
    test_right_inverse<Layout<Shape<_16, Shape<_8, _8>>, Stride<_8, Stride<_128, _1>>>>(ps);
    // test_right_inverse<Layout<Shape<Shape<_3,_2>,Shape<_4,_2>>,Stride<Stride<_4,_1>,Stride<_12,_2>>>>(ps);
    // test_right_inverse<Layout<Shape<_32,_8>,Stride<_1,_32>>>(ps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Composition
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename B>
void test_composition(int ps) {
    // auto res = composition(A{}, B{});
    auto res = A{}.compose(B{}, _);

    // clang-format off
    print("Input  : ");custom_print(A{},ps);print(" , ");custom_print(B{},ps);print("\n");
    print("Output : ");custom_print(res,ps);print("\n");
    // clang-format on
}

void test_composition_examples(int ps) {

    // test_composition<
    //   Layout<Shape<Int<20>, _2>, Stride<_16, _4>>,
    //   Layout<Shape<_4, _5>, Stride<_1, _4>>>(ps);

    test_composition<Layout<Shape<Shape<_2, _4>, Shape<_3, _5>>>, Layout<Shape<_1, _2>>>(ps);

    // test_composition<
    //   Layout<Shape<_2, _32>, Stride<_32, _1>>,
    //   Layout<Shape<Shape<_8, _4>, _8>, Stride<Stride<_8, _0>, _1>>>(ps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                Right Inverse of Ref then Compose to Src
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename LayoutRef, typename LayoutSrc>
void test_RIRCS(int ps) {
    auto res = right_inverse(LayoutRef{}).compose(LayoutSrc{});

    // clang-format off
    print("Reference Layout : ");custom_print(LayoutRef{},ps);print("\n");
    print("Source    Layout : ");custom_print(LayoutSrc{},ps);print("\n");
    print("Result    Layout : ");custom_print(res,ps);print("\n");
    // clang-format on
}

void test_RIRCS_examples(int ps) {
    {
        print("Source : 128b , Val : 16b \n");
        using copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;
        print(copy_atom{});
        test_RIRCS<typename copy_atom::ValLayoutRef, typename copy_atom::ValLayoutSrc>(ps);
        print("\n");
    }

    {
        print("Source : 128b , Val : 8b \n");
        using copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, int8_t>;
        print(copy_atom{});
        test_RIRCS<typename copy_atom::ValLayoutRef, typename copy_atom::ValLayoutSrc>(ps);
        print("\n");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Rank
// Number of mode in layout is rank.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
void test_rank() {
    auto res = rank_v<Layout>;
    print("Rank of Layout : ");
    print(Layout{});
    print(" is : ");
    print(res);
    print("\n");
}

void test_rank_examples() {
    test_rank<Layout<Shape<_1>>>();
    test_rank<Layout<Shape<_16, _8>, Stride<_8, _1>>>();
    test_rank<Layout<Shape<_1, _8>>>();
    test_rank<Layout<Shape<Shape<_2, _4>, _2>>>();
    test_rank<Layout<Shape<Shape<_2, _4>, _2, _1>>>();
    test_rank<Layout<Shape<Shape<_2, _4>, _2, _0>>>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Append
// If N is less than size tuple throw error
// if N is equal to size of tuple return same tuple
// if N is great than size of tuple then append up to N.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename X, int N>
void test_append() {
    auto res = append<N>(T{}, X{});

    // clang-format off
    print("Input  : ");print(T{});print(" , ");print(X{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_append_examples() {
    // {
    //     print("Append at 1st position \n");
    //     test_append<Layout<Shape<_16, _8>, Stride<_8, _1>>,
    //                 Layout<_1>,
    //                 1>();
    //     test_append<Layout<Shape<_1, _8>>,
    //                 Layout<_1>,
    //                 1>();
    // }
    {
        print("Append at 2nd position \n");
        test_append<Layout<Shape<_16, _8>, Stride<_8, _1>>, Layout<_1>, 2>();
        test_append<Layout<Shape<_1, _8>>, Layout<_1>, 2>();
    }

    {
        print("Append at 3rd position \n");
        test_append<Layout<Shape<_16, _8>, Stride<_8, _1>>, Layout<_1>, 3>();
        test_append<Layout<Shape<_1, _8>>, Layout<_1>, 3>();
    }
    {
        print("Append at 4th position \n");
        test_append<Layout<Shape<_16, _8>, Stride<_8, _1>>, Layout<_2>, 4>();
        test_append<Layout<Shape<_1, _8>>, Layout<_2>, 4>();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Repeat
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
void test_repeat() {
    auto res = repeat<N>(_);

    // clang-format off
    print(" Output : ");print(res);print("\n");
    // clang-format on
}

void test_repeat_examples() {
    test_repeat<1>();
    test_repeat<2>();
    test_repeat<3>();
    test_repeat<4>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                          Raked product
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tile, typename MatOfTiles>
void test_raked_product(int ps) {
    auto res = raked_product(Tile{}, MatOfTiles{});

    // clang-format off
    print("Input  : ");custom_print(Tile{},ps);print(" , ");custom_print(MatOfTiles{},ps);print("\n");
    print("Output : ");custom_print(res,ps);print("\n");
    // clang-format on
}

void test_raked_product_examples(int ps) {
    test_raked_product<Layout<Shape<_16, _8>, Stride<_8, _1>>, Layout<Shape<_1, _8>>>(ps);
    // test_raked_product<Layout<Shape<_2, _2>>, Layout<Shape<_3, _4>>>(ps);
    // test_raked_product<Layout<Shape<_32, _1>>, Layout<Shape<_1, _8>>>(ps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                          With Shape
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout, typename Shape>
void test_with_shape(int ps) {
    auto res = Layout{}.with_shape(Shape{});

    // clang-format off
    print("Input  : ");print(Layout{});print(" , ");print(Shape{});print("\n");
    print("Output : ");custom_print(res,ps);print("\n");
    // clang-format on
}

void test_with_shape_examples(int ps) {
    test_with_shape<Layout<Shape<_8, _128>, Stride<_128, _1>>, Shape<_128, _8>>(ps);
    // test_with_shape<Layout<Shape<_2,_2,_3,_4>,Stride<_3,_24,_1,_6>>,Shape<_4,_12>>(ps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                          zipped product
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout, typename Tile>
void test_zipped_product(int ps) {
    auto res = zipped_product(Layout{}, Tile{});

    // clang-format off
    print("Input : ");print(Layout{});print(" , ");print(Tile{});print("\n");
    custom_print(res,ps);print("\n");
    // clang-format on
}

void test_zipped_product_examples(int ps) {
    test_zipped_product<Layout<Shape<_2, _2>>, Layout<Shape<_2, _3>>>(ps);
    test_zipped_product<
      Layout<Shape<_2, _3>, Stride<_3, _1>>,
      Layout<Shape<_2, _2>, Stride<_2, _1>>>(ps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                          zipped divide
// Layout   ==> thr,val
// tile     ==> g_thr,g_val  gather
// zip_div  ==> (g_thr,g_val),(r_thr,r_val) reminder

// Layout   ==> ((_2,_3),(_4,_6)):((_1,_2),(_6,_24))
// tile     ==> (_1,_8)
// zip_div  ==> ((_1,_8),(_6,_3)):((_0,_6),(_1,_48))
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout, typename Tile>
void test_zipped_divide(int ps) {
    auto res = zipped_divide(Layout{}, Tile{});

    // clang-format off
    print("Input : ");custom_print(Layout{},ps);print(" , ");print(Tile{});print("\n");
    custom_print(res,ps);print("\n");
    // clang-format on
}

void test_zipped_divide_examples(int ps) {
    test_zipped_divide<Layout<Shape<Shape<_2, _3>, Shape<_4, _6>>>, Shape<_1, _8>>(ps);
    test_zipped_divide<
      Layout<Shape<Shape<_2, _3>, Shape<_4, _6>>, Stride<Stride<_4, Int<48>>, Stride<_1, _8>>>,
      Shape<_1, _8>>(ps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                          tiled product
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout, typename Tile>
void test_tiled_product() {
    auto res    = tiled_product(Layout{}, Tile{});
    auto tensor = make_counting_tensor(res);

    // clang-format off
    print("Input : ");print(Layout{});print(" , ");print(Tile{});print("\n");
    print_tensor(tensor);print("\n\n");
    // clang-format on
}

void test_tiled_product_examples() {
    test_tiled_product<Layout<Shape<_2, _2>>, Layout<Shape<_2, _3>>>();
    test_tiled_product<
      Layout<Shape<_2, _3>, Stride<_3, _1>>,
      Layout<Shape<_2, _2>, Stride<_2, _1>>>();
    test_tiled_product<Layout<_32>, Layout<Shape<_2, _2, _1>, Stride<_1, _2, _0>>>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//  *                      max common vector
//  ? Return Int<N> such that N is the maximum number of contiguous elements
//  ? that logically correspond in the layouts of @a a and @a b. This is,
//  ? the number of elements that could reasonably be "vectorized" in the layouts.

// Examples
// Input  : (_4,_4):(_1,_4) , (_2,_2):(_1,_2)
// Output : _4

// Input  : (_2,_2):(_1,_2) , (_4,_4):(_1,_4)
// Output : _16

// Input  : (_4,_4):(_4,_1) , (_2,_2):(_2,_1)
// Output : _1

// Input  : (_4,_4):(_1,_8) , (_2,_2):(_1,_4)
// Output : _2

// Input  : _4:_2 , _4:_2
// Output : _1
// Even shape and stride are same this not contiguous in memory layout
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcLayout, typename DstLayout>
void test_max_common_vector() {
    auto res = max_common_vector(SrcLayout{}, DstLayout{});

    // clang-format off
    print("Input  : ");print(SrcLayout{});print(" , ");print(DstLayout{});print("\n");
    print("Output : ");print(res);print("\n\n");
    // clang-format on
}

void test_max_common_vector_examples() {
    // col major
    test_max_common_vector<Layout<Shape<_4, _4>>, Layout<Shape<_2, _2>>>();
    test_max_common_vector<Layout<Shape<_2, _2>>, Layout<Shape<_4, _4>>>();

    // row  major
    test_max_common_vector<
      Layout<Shape<_4, _4>, Stride<_4, _1>>,
      Layout<Shape<_2, _2>, Stride<_2, _1>>>();
    test_max_common_vector<
      Layout<Shape<_2, _2>, Stride<_2, _1>>,
      Layout<Shape<_4, _4>, Stride<_4, _1>>>();

    // row and col major
    test_max_common_vector<
      Layout<Shape<_4, _4>, Stride<_4, _1>>,
      Layout<Shape<_2, _2>, Stride<_1, _2>>>();
    test_max_common_vector<
      Layout<Shape<_2, _2>, Stride<_2, _1>>,
      Layout<Shape<_4, _4>, Stride<_1, _4>>>();

    // stride
    test_max_common_vector<
      Layout<Shape<_4, _4>, Stride<_1, _8>>,
      Layout<Shape<_2, _2>, Stride<_1, _4>>>();
    test_max_common_vector<
      Layout<Shape<_2, _2>, Stride<_1, _4>>,
      Layout<Shape<_4, _4>, Stride<_1, _8>>>();

    test_max_common_vector<Layout<_4, _2>, Layout<_4, _2>>();

    test_max_common_vector<
      Layout<Shape<Shape<_8, _1>, _1, _1>, Stride<Stride<_1, _0>, _0, _0>>,
      Layout<Shape<Shape<_8, _1>, _1, _1>, Stride<Stride<_1, _0>, _0, _0>>>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Thread Layout and Val Layout
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ThrLayout, typename ValLayout>
void test_build_layoutTV(std::string test_name) {
    auto interleaved    = raked_product(ThrLayout{}, ValLayout{});
    auto val_thr_layout = right_inverse(interleaved);
    auto thr_val_layout =
      val_thr_layout.with_shape(make_shape(size(ThrLayout{}), size(ValLayout{})));
    auto mn_layout = make_layout(product_each(shape(interleaved)));
    auto zip_div   = zipped_divide(mn_layout, shape(ValLayout{}));

    print_latex(mn_layout, (test_name + std::string("_mn")).c_str());
    print_latex(zip_div, (test_name + std::string("_zip_div")).c_str());
    print_latex(interleaved, (test_name + std::string("_interleaved")).c_str());
    // clang-format off
    print("%% val_thr_layout : ");print(val_thr_layout);
    // clang-format on
    print_latex(thr_val_layout, (test_name + std::string("_thr_val_layout")).c_str());
}

void test_build_layoutTV_examples() {
    print_latex_header();
    test_build_layoutTV<Layout<Shape<_2, _3>>, Layout<Shape<_4, _5>>>("T2x3_V4x5");
    print_latex_footer();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  TidFrag
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ThrLayout, typename ValLayout>
void test_tidFrag(std::string test_name) {
    using tiled_copy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{}, ThrLayout{}, ValLayout{}));

    auto mn_layout =
      make_layout(product_each(zip((typename tiled_copy::TiledShape_MN{}), make_tuple(2, 2))));

    auto res        = tiled_copy::tidfrg_S(mn_layout);
    auto res_tensor = make_counting_tensor(res);

    // print_latex(mn_layout, (test_name + "_mn_layout").c_str());
    // print_latex(res, (test_name + "_tidfrg_S").c_str());
    print(mn_layout);
    print("\n");
    print(res);
    print("\n");
    print(coalesce(res));
    print_tensor(res_tensor);
}

void test_tidFrag_examples() {
    // print_latex_header();
    test_tidFrag<Layout<Shape<_2, _3>>, Layout<Shape<_4, _6>>>("tidFrag_T2x3_V4x6");
    // print_latex_footer();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  Tile to Thread Frag
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename CopyAtom, typename ThrLayout, typename ValLayout, typename RestShape_MN>
void test_tile_thrFrag(std::string test_name, int ps) {
    using tiled_copy = decltype(make_tiled_copy(CopyAtom{}, ThrLayout{}, ValLayout{}));

    auto thr_copy = tiled_copy{}.get_thread_slice(1);

    auto mn_layout =
      make_layout(product_each(zip((typename tiled_copy::TiledShape_MN{}), RestShape_MN{})));

    auto mn_tensor = make_counting_tensor(mn_layout);

    // auto mn_layout = make_layout(typename tiled_copy::TiledShape_MN{});

    auto tile    = zipped_divide(mn_layout, typename tiled_copy::Tiler_MN{});
    auto ref2trg = right_inverse(typename tiled_copy::AtomLayoutRef{})
                     .compose(typename tiled_copy::AtomLayoutDst{});

    auto atom_layout_TV = zipped_divide(
      typename tiled_copy::TiledLayout_TV{},
      make_shape(typename tiled_copy::AtomNumThr{}, typename tiled_copy::AtomNumVal{}));

    auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
    auto thrval2mn     = coalesce(zip(trg_layout_TV), Shape<_1, Shape<_1, _1>>{});
    auto tv_tensor     = tile.compose(thrval2mn, _);
    auto res           = tv_tensor(make_coord(_, _), _);
    auto res_tensor    = make_counting_tensor(res);
    auto part          = thr_copy.partition_S(mn_tensor);

    // clang-format off
    print("%%  AtomThrID      : ");print       (typename tiled_copy::AtomThrID      {}                                         );print("\n");
    print("%%  AtomLayoutSrc  : ");custom_print(typename tiled_copy::AtomLayoutSrc  {},(test_name+"_AtomLayoutSrc").c_str(),ps );print("\n");
    print("%%  AtomLayoutDst  : ");custom_print(typename tiled_copy::AtomLayoutDst  {},(test_name+"_AtomLayoutDst").c_str(),ps );print("\n");
    print("%%  AtomLayoutRef  : ");custom_print(typename tiled_copy::AtomLayoutRef  {},(test_name+"_AtomLayoutRef").c_str(),ps );print("\n");
    print("%%  AtomNumThr     : ");print       (typename tiled_copy::AtomNumThr     {}                                         );print("\n");
    print("%%  AtomNumVal     : ");print       (typename tiled_copy::AtomNumVal     {}                                         );print("\n");
    print("%%  Tiler_MN       : ");print       (typename tiled_copy::Tiler_MN       {}                                         );print("\n");
    print("%%  TiledShape_MN  : ");print       (typename tiled_copy::TiledShape_MN  {}                                         );print("\n");
    print("%%  TiledLayout_TV : ");custom_print(typename tiled_copy::TiledLayout_TV {},(test_name+"_TiledLayout_TV").c_str(),ps);print("\n");
    print("%%  TiledNumThr    : ");print       (typename tiled_copy::TiledNumThr    {}                                         );print("\n");
    print("%%  TiledNumVal    : ");print       (typename tiled_copy::TiledNumVal    {}                                         );print("\n");
    if(ps==1)
        {print("%%  TiledCopy      : ");print_latex(         tiled_copy                 {},(test_name+"_TiledCopy").c_str()     );print("\n");}


    print("%%  Mn_Layout      : ");custom_print(mn_layout      ,(test_name+"_Mn_Layout"      ).c_str(),ps );print("\n");
    print("%%  Tile           : ");custom_print(tile           ,(test_name+"_Tile"           ).c_str(),ps );print("\n");
    print("%%  Ref2trg        : ");custom_print(ref2trg        ,(test_name+"_Ref2trg"        ).c_str(),ps );print("\n");
    print("%%  Atom_Layout_Tv : ");custom_print(atom_layout_TV ,(test_name+"_Atom_Layout_Tv" ).c_str(),ps );print("\n");
    print("%%  Trg_Layout_Tv  : ");custom_print(trg_layout_TV  ,(test_name+"_Trg_Layout_Tv"  ).c_str(),ps );print("\n");
    print("%%  Thrval2mn      : ");custom_print(thrval2mn      ,(test_name+"_Thrval2mn"      ).c_str(),ps );print("\n");
    print("%%  Tv_Tensor      : ");custom_print(tv_tensor      ,(test_name+"_Tv_Tensor"      ).c_str(),ps );print("\n");
    
    print("%%  Tile2Frag      : " );print       (res        );print("\n");
    if(ps !=1)
        {
        print("%%  TiledCopy      : " );print       ( tiled_copy {} );print("\n");
        print("%% Tile2Frag       : " );print_tensor( res_tensor    );print("\n");
        print("%% Partition       : " );print_tensor( part          );print("\n");
        }
    // clang-format on
}

void test_tile_thrFrag_examples(int ps) {
    if (ps == 1)
        print_latex_header();
    // test_tile_thrFrag<
    //   Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint32_t>, half_t>,
    //   Layout<Shape<_2, _2>>,
    //   Layout<Shape<_2, _2>>,
    //   Shape<_2, _1>>("tile2Frag_T2x2_V2x2", ps);
    // test_tile_thrFrag<
    //   Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>,
    //   Layout<Shape<_2, _3>>,
    //   Layout<Shape<_4, _6>>,
    //   Shape<_2, _2>>("tile2Frag_T2x3_V4x6", ps);
    test_tile_thrFrag<
      Copy_Atom<SM75_U32x1_LDSM_N, half_t>,
      Layout<Shape<_4, _8>>,
      Layout<Shape<_2, _1>>,
      Shape<_2, _2>>("tile2Frag_LDSM_T4x8_V4x2", ps);
    if (ps == 1)
        print_latex_footer();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                              MMa thr Frag
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Operation,
  typename AtomLayoutMNK   = Layout<Shape<_1, _1, _1>>,
  typename ValLayoutMNK    = Layout<Shape<_1, _1, _1>>,
  typename PermutationsMNK = Tile<Underscore, Underscore, Underscore>>
void test_mma_thr_Frag(std::string test_name, int ps) {
    using tiled_mma = TiledMMA<MMA_Atom<Operation>, AtomLayoutMNK, ValLayoutMNK, PermutationsMNK>;

    // clang-format off
    print("%%  AtomShape_MNK   : ");print       (typename tiled_mma::AtomShape_MNK  {}                                         );print("\n");
    print("%%  AtomLayoutC_TV  : ");custom_print(typename tiled_mma::AtomLayoutC_TV {},(test_name+"_AtomLayoutC_TV").c_str(),ps);print("\n");
    print("%%  AtomLayoutA_TV  : ");custom_print(typename tiled_mma::AtomLayoutA_TV {},(test_name+"_AtomLayoutA_TV").c_str(),ps);print("\n");
    print("%%  AtomLayoutB_TV  : ");custom_print(typename tiled_mma::AtomLayoutB_TV {},(test_name+"_AtomLayoutB_TV").c_str(),ps);print("\n");
    print("%%  AtomThrID       : ");print       (typename tiled_mma::AtomThrID      {}                                         );print("\n");
    print("%%  TiledShape_MNK  : ");print       (typename tiled_mma::TiledShape_MNK {}                                         );print("\n");
    print("%%  ThrLayoutVMNK   : ");print       (typename tiled_mma::ThrLayoutVMNK  {}                                         );print("\n");
    print("%%  TidLayout       : ");print       (typename tiled_mma::TidLayout      {}                                         );print("\n");
    // clang-format on

    auto ref = make_layout(make_shape(
      size<0>(typename tiled_mma::TiledShape_MNK{}),
      size<1>(typename tiled_mma::TiledShape_MNK{})));
    auto t_tile =
      make_tile(left_inverse(get<0>(PermutationsMNK{})), left_inverse(get<1>(PermutationsMNK{})));
    auto t_tensor = logical_divide(ref, t_tile);
    auto a_tile   = make_tile(
      make_layout(size<0>(typename tiled_mma::AtomShape_MNK{})),
      make_layout(size<1>(typename tiled_mma::AtomShape_MNK{})));
    auto a_tensor  = zipped_divide(t_tensor, a_tile);
    auto tv_tensor = a_tensor.compose(typename tiled_mma::AtomLayoutC_TV{}, _);
    auto thr_tile  = make_tile(
      _,
      make_tile(
        make_layout(size<1>(typename tiled_mma::ThrLayoutVMNK{})),
        make_layout(size<2>(typename tiled_mma::ThrLayoutVMNK{}))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
    auto tid_frag   = thr_tensor.compose(typename tiled_mma::TidLayout{}, _);

    // clang-format off
    print("%%  Ref        : ");custom_print (ref        ,(test_name+"_Ref").c_str(),ps       );print("\n");
    print("%%  T_Tile     : ");print        (t_tile                                          );print("\n");
    print("%%  T_Tensor   : ");custom_print (t_tensor   ,(test_name+"_T_Tensor").c_str(),ps  );print("\n");
    print("%%  A_Tile     : ");print        (a_tile                                          );print("\n");
    print("%%  A_Tensor   : ");custom_print (a_tensor   ,(test_name+"_A_Tensor").c_str(),ps  );print("\n");
    print("%%  Tv_Tensor  : ");custom_print (tv_tensor  ,(test_name+"_Tv_Tensor").c_str(),ps );print("\n");
    print("%%  Thr_Tile   : ");print        (thr_tile                                        );print("\n");
    print("%%  Thr_Tensor : ");custom_print (thr_tensor,(test_name+"_Thr_Tensor").c_str(),ps );print("\n");
    print("%%  Tid_Frag   : ");custom_print (tid_frag   ,(test_name+"_Tid_Frag").c_str(),ps  );print("\n");
    // clang-format on
}

void test_mma_thr_Frag_examples(int ps) {
    if (ps == 1)
        print_latex_header();
    test_mma_thr_Frag<SM80_16x8x16_F16F16F16F16_TN>("mma_m16n8k16_f16f16f16f16", ps);
    if (ps == 1)
        print_latex_footer();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  Data and Thread Arrangement for mma
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename MMA_Atom_OP_, typename Atom_Layout, typename Val_Layout>
void test_data_and_thread_arrangement_for_mma(std::string test_name) {
    auto tiled_mma = TiledMMA<MMA_Atom<MMA_Atom_OP_>, Atom_Layout, Val_Layout>{};
    print_latex(tiled_mma, ("TiledMMA_" + test_name).c_str());
}

void test_data_and_thread_arrangement_for_mma_examples() {
    print_latex_header();
    test_data_and_thread_arrangement_for_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_1, _1, _1>>,
      Layout<Shape<_1, _1, _1>>>("A1x1x1_V1x1x1");
    test_data_and_thread_arrangement_for_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_2, _2, _1>>,
      Layout<Shape<_1, _1, _1>>>("A2x2x1_V1x1x1");
    test_data_and_thread_arrangement_for_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_2, _2, _1>>,
      Layout<Shape<_1, _2, _1>>>("A2x2x1_V1x2x1");
    print_latex_footer();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  Tiled MMA partition the SharedMemory

// Examples 1:
// Smem Sh :  (128,128,32)
// TiledThr:  (_1,_1,_1):(_0,_0,_0)
// Atom_sh :  (16,8,8)
// th      :  (16,8,8)
//*smem/th :  (8,16,4)

// TiledVal:  (_1,_1,_1):(_0,_0,_0)
// *           (Atm_val,smem/th)
// PART_A   : ((_2,_2),_8,_4):((_8,_512),_16,_1024)
// PART_B   : (_2,_16,_4):(_512,_8,_1024)

// Examples 2:
// Smem Sh :  (128,128,32)
// TiledThr:  (_2,_1,_1):(_1,_0,_0)
// Atom_sh :  (16,8,8)
// th      :  (32,8,8)
//*smem/th :  (4,16,4)
// TiledVal:  (_1,_1,_1):(_0,_0,_0)

// PART_A   : ((_2,_2),_4,_4):((_8,_512),_32,_1024)
// PART_B   : (_2,_16,_4):(_512,_8,_1024)

// Examples 3:
// Smem Sh :  (128,128,32)
// TiledThr:  (_2,_1,_1):(_1,_0,_0)
// Atom_sh :  (16,8,8)
// th      :  (32,8,8)
//*smem/th :  (4,16,4)
// TiledVal:  (_2,_1,_1):(_1,_0,_0)

// PART_A   : ((_2,_2),_4,_4):((_8,_512),_32,_1024)
// PART_B   : (_2,_16,_4):(_512,_8,_1024)

// Examples 4:
// Smem Sh :  (128,128,32)
// TiledThr:  (_2,_2,_1):(_1,_2,_0)
// Atom_sh :  (16,8,8)
// th      :  (32,16,8)
//*smem/th :  (4,8,4)
// TiledVal:  (_1,_1,_1):(_0,_0,_0)

// PART_A   : ((_2,_2),_4,_4):((_8,_512),_32,_1024)
// PART_B   : (_2,_8,_4):(_512,_16,_1024)

// Examples 5:
// Smem Sh :  (128,128,32)
// TiledThr:  (_2,_2,_1):(_1,_2,_0)
// Atom_sh :  (16,8,8)
// th      :  (32,16,8)
//*smem/th :  (4,8,4)
// TiledVal:  (_2,_2,_1):(_1,_2,_0)

// PART_A   : ((_2,_2),_4,_4):((_8,_512),_32,_1024)
// PART_B   : (_2,_8,_4):(_512,_16,_1024)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename MMA_Atom_Op, typename Atom_Layout, typename Val_Layout, typename Smem_ShapeMNK>
void test_partition_smem_by_tiled_mma() {

    auto smem_tensorA = make_counting_tensor(
      make_layout(make_shape(get<0>(Smem_ShapeMNK{}), get<2>(Smem_ShapeMNK{}))));
    auto smem_tensorB = make_counting_tensor(
      make_layout(make_shape(get<1>(Smem_ShapeMNK{}), get<2>(Smem_ShapeMNK{}))));
    auto tiled_mma = TiledMMA<MMA_Atom<MMA_Atom_Op>, Atom_Layout, Val_Layout>{};
    auto thr_mma   = tiled_mma.get_thread_slice(0);
    auto partA     = thr_mma.partition_A(smem_tensorA);
    auto partB     = thr_mma.partition_B(smem_tensorB);

    // clang-format off
    print("*******************************************\n");
    print("SMEM_TENSOR_A : ");print (smem_tensorA );print("\n");
    print("SMEM_TENSOR_B : ");print (smem_tensorB );print("\n");
    print("TILED_MMA     : ");print (tiled_mma    );print("\n");
    print("PART_A        : ");print (partA        );print("\n");
    print("PART_B        : ");print (partB        );print("\n");
    print("*******************************************\n");
    // print_latex(Smem_Layout{},"Layout128x32",3);
    // clang-format on
}

void test_partition_smem_by_tiled_mma_examples() {
    test_partition_smem_by_tiled_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_1, _1, _1>>,
      Layout<Shape<_1, _1, _1>>,
      Shape<_128, _128, _32>>();
    test_partition_smem_by_tiled_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_2, _1, _1>>,
      Layout<Shape<_1, _1, _1>>,
      Shape<_128, _128, _32>>();
    test_partition_smem_by_tiled_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_2, _1, _1>>,
      Layout<Shape<_2, _1, _1>>,
      Shape<_128, _128, _32>>();
    test_partition_smem_by_tiled_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_2, _2, _1>>,
      Layout<Shape<_1, _1, _1>>,
      Shape<_128, _128, _32>>();
    test_partition_smem_by_tiled_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_2, _2, _1>>,
      Layout<Shape<_2, _2, _1>>,
      Shape<_128, _128, _32>>();
    test_partition_smem_by_tiled_mma<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_2, _2, _1>>,
      Layout<Shape<_1, _2, _1>>,
      Shape<_128, _128, _32>>();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  Usage of Val layout in the Thread layout TV --> Copy Atom

// Examples 1:

// TiledThr:  (_1,_1,_1):(_0,_0,_0)
// TiledVal:  (_1,_1,_1):(_0,_0,_0)

//*                           (atom_thr,atom_val)
// MMA_Atom LayoutA_TV:    ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
//*                          (atom_thr,tiled_thr),(val,tiled_val)
// Copy_Atm LAYOUT_A_TV : ((_4,_8),((_2,_2),(_1,_1))):((_16,_1),((_8,_64),(_0,_0)))

// MMA_ATOM LayoutB_TV:    ((_4,_8),_2):((_8,_1),_32)
// Copy_Atm LAYOUT_B_TV : ((_4,_8),(_2,(_1,_1))):((_8,_1),(_32,(_0,_0)))

// MMA_ATOM LayoutC_TV:    ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
// Copy_Atom LAYOUT_C_TV : ((_4,_8),((_2,_2),(_1,_1))):((_32,_1),((_16,_8),(_0,_0)))

// Examples 2:

// TiledThr:  (_2,_1,_1):(_1,_0,_0)
// TiledVal:  (_1,_1,_1):(_0,_0,_0)

//*                           (atom_thr,atom_val)
// MMA_Atom LayoutA_TV:    ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
//*                          (atom_thr,tiled_thr),(val,tiled_val)
// Copy_Atm LAYOUT_A_TV : ((_4,_8,_2),((_2,_2),(_1,_1))):((_32,_1,_16),((_8,_128),(_0,_0)))

// MMA_ATOM LayoutB_TV:    ((_4,_8),_2):((_8,_1),_32)
// Copy_Atm LAYOUT_B_TV : ((_4,_8,_2),(_2,(_1,_1))):((_8,_1,_0),(_32,(_0,_0)))

// MMA_ATOM LayoutC_TV:    ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
// Copy_Atom LAYOUT_C_TV : ((_4,_8,_2),((_2,_2),(_1,_1))):((_64,_1,_16),((_32,_8),(_0,_0)))

// Examples 3:

// TiledThr:  (_2,_1,_1):(_1,_0,_0)
// TiledVal:  (_2,_1,_1):(_1,_0,_0)

// MMA_Atom LayoutA_TV:    ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
// Copy_Atm LAYOUT_A_TV : ((_4,_8,_2),((_2,_2),(_2,_1))):((_64,_1,_16),((_8,_256),(_32,_0)))

// MMA_ATOM LayoutB_TV:    ((_4,_8),_2):((_8,_1),_32)
// Copy_Atm LAYOUT_B_TV : ((_4,_8,_2),(_2,(_1,_1))):((_8,_1,_0),(_32,(_0,_0)))

// MMA_ATOM LayoutC_TV:    ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
// Copy_Atom LAYOUT_C_TV : ((_4,_8,_2),((_2,_2),(_2,_1))):((_128,_1,_16),((_64,_8),(_32,_0)))

// examples 4:

//   TiledThr:  (_2,_2,_1):(_1,_2,_0)
//   TiledVal:  (_1,_1,_1):(_0,_0,_0)

//   MMA_Atom LayoutA_TV:    ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
//   Copy_Atm LAYOUT_A_TV : ((_4,_8,_2,_2),((_2,_2),(_1,_1))):((_32,_1,_16,_0),((_8,_128),(_0,_0)))

//   MMA_ATOM LayoutB_TV:    ((_4,_8),_2):((_8,_1),_32)
//   Copy_Atm LAYOUT_B_TV : ((_4,_8,_2,_2),(_2,(_1,_1))):((_16,_1,_0,_8),(_64,(_0,_0)))

//   MMA_ATOM LayoutC_TV:    ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
//   Copy_Atom LAYOUT_C_TV :
//   ((_4,_8,_2,_2),((_2,_2),(_1,_1))):((_64,_1,_16,_256),((_32,_8),(_0,_0)))

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename MMA_Atom_Op, typename Atom_Layout, typename Val_Layout, typename Smem_ShapeMNK>
void test_usage_val_layout_in_TV_layout(std::string test_name) {

    auto ivec = std::vector<int>(128 * 32);
    std::iota(ivec.begin(), ivec.end(), 0);
    auto smem_tensorA = make_tensor(
      ivec.data(),
      tile_to_shape(
        composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}),
        make_shape(get<0>(Smem_ShapeMNK{}), get<2>(Smem_ShapeMNK{}))));
    auto smem_tensorB = make_counting_tensor(
      make_layout(make_shape(get<1>(Smem_ShapeMNK{}), get<2>(Smem_ShapeMNK{}))));
    auto tiled_mma  = TiledMMA<MMA_Atom<MMA_Atom_Op>, Atom_Layout, Val_Layout>{};
    auto layoutA_TV = tiled_mma.get_layoutA_TV();
    auto layoutB_TV = tiled_mma.get_layoutB_TV();
    auto layoutC_TV = tiled_mma.get_layoutC_TV();

    auto sr_copy_atom    = Copy_Atom<SM75_U32x4_LDSM_N, tfloat32_t>{};
    auto sr_tiled_copy_A = make_tiled_copy_A(sr_copy_atom, tiled_mma);
    auto sr_tiled_copy_B = make_tiled_copy_B(sr_copy_atom, tiled_mma);

    auto sr_layoutA_TV = sr_tiled_copy_A.get_layoutS_TV();
    auto sr_layoutB_TV = sr_tiled_copy_B.get_layoutS_TV();

    auto thr_copyA  = sr_tiled_copy_A.get_thread_slice(0);
    auto copy_partA = thr_copyA.partition_S(smem_tensorA);

    auto thr_mmaA  = tiled_mma.get_thread_slice(0);
    auto mma_partA = thr_mmaA.partition_A(smem_tensorA);

    // clang-format off
    print("%% *******************************************\n");
    // print("%% TILED_MMA      : " );print_latex(tiled_mma     ,("val_lay_usage__"+test_name+"_TILED_MMA"    ).c_str());print("\n");
    // print("%% LAYOUT_A_TV    : " );print_latex(layoutA_TV    ,("val_lay_usage__"+test_name+"_LAYOUT_A_TV"  ).c_str());print("\n");
    // print("%% LAYOUT_B_TV    : " );print_latex(layoutB_TV    ,("val_lay_usage__"+test_name+"_LAYOUT_B_TV"  ).c_str());print("\n");
    // print("%% LAYOUT_C_TV    : " );print_latex(layoutC_TV    ,("val_lay_usage__"+test_name+"_LAYOUT_C_TV"  ).c_str());print("\n");
    // print("%% sr_layoutA_TV  : " );print_latex(sr_layoutA_TV ,("val_lay_usage__"+test_name+"_sr_layoutA_TV").c_str());print("\n");
    print("%% smem_layoutA   : " );print_latex(smem_tensorA.layout()  ,("val_lay_usage__"+test_name+"_smem_layoutA" ).c_str());print("\n");
    // print("%% sr_layoutB_TV  : " );print_latex(sr_layoutB_TV ,("val_lay_usage__"+test_name+"_sr_layoutB_TV").c_str());print("\n");
    // print("%% copy Part A : ");print_tensor(copy_partA);print("\n");
    // print("%% MMa Part A : ");print_tensor(mma_partA);print("\n");
    print("%% *******************************************\n");
    // clang-format on
}

void test_usage_val_layout_in_TV_layout_examples() {
    print_latex_header();
    // test_usage_val_layout_in_TV_layout<
    //   SM80_16x8x8_F32TF32TF32F32_TN,
    //   Layout<Shape<_1, _1, _1>>,
    //   Layout<Shape<_1, _1, _1>>,
    //   Shape<_128, _128, _32>>("A1x1x1_V1x1x1");
    // test_usage_val_layout_in_TV_layout<
    //   SM80_16x8x8_F32TF32TF32F32_TN,
    //   Layout<Shape<_2, _1, _1>>,
    //   Layout<Shape<_1, _1, _1>>,
    //   Shape<_128, _128, _32>>("A2x1x1_V1x1x1");
    // test_usage_val_layout_in_TV_layout<
    //   SM80_16x8x8_F32TF32TF32F32_TN,
    //   Layout<Shape<_2, _1, _1>>,
    //   Layout<Shape<_2, _1, _1>>,
    //   Shape<_128, _128, _32>>("A2x1x1_V2x1x1");
    // test_usage_val_layout_in_TV_layout<
    //   SM80_16x8x8_F32TF32TF32F32_TN,
    //   Layout<Shape<_2, _2, _1>>,
    //   Layout<Shape<_1, _1, _1>>,
    //   Shape<_128, _128, _32>>("A2x2x1_V1x1x1");
    // test_usage_val_layout_in_TV_layout<
    //   SM80_16x8x8_F32TF32TF32F32_TN,
    //   Layout<Shape<_2, _2, _1>>,
    //   Layout<Shape<_2, _2, _1>>,
    //   Shape<_128, _128, _32>>("A2x2x1_V2x2x1");
    test_usage_val_layout_in_TV_layout<
      SM80_16x8x8_F32TF32TF32F32_TN,
      Layout<Shape<_2, _2, _1>>,
      Layout<Shape<_1, _2, _1>>,
      Shape<_128, _128, _32>>("A2x2x1_V1x2x1");
    print_latex_footer();
}

int main(int argc, char *argv[]) {
    // print_select
    [[maybe_unused]] int ps{-1};
    if (argc >= 2)
        ps = atoi(argv[1]);

    // test_transform_leaf_examples();
    // test_transform_examples();
    // test_find_examples();
    // test_find_if_examples();
    // test_compact_col_major_examples();
    // test_inverse_seq_examples();
    // test_right_inverse_examples(ps);
    // test_composition_examples(ps);
    // test_RIRCS_examples(ps);
    // test_repeat_examples();
    // test_rank_examples();
    // test_append_examples();
    // test_raked_product_examples(ps);
    // test_with_shape_examples(ps);
    // test_zipped_product_examples(ps);
    // test_zipped_divide_examples(ps);
    // test_tiled_product_examples();
    // test_max_common_vector_examples();
    // test_build_layoutTV_examples();
    // test_tidFrag_examples();
    // test_tile_thrFrag_examples(ps);
    // test_mma_thr_Frag_examples(ps);
    // test_data_and_thread_arrangement_for_mma_examples();
    // test_partition_smem_by_tiled_mma_examples();
    test_usage_val_layout_in_TV_layout_examples();
}