#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

template <typename Args>
void custom_print(Args args, int ps = -1)
{
    switch (ps)
    {
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
void test_transform_leaf()
{
    auto res = transform_leaf(Tuple{}, Fn{});

    // clang-format off
    print("Input  : ");print(Tuple{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_transform_leaf_examples()
{
    test_transform_leaf<tuple<_2, _m1, tuple<_m5, _1>>, abs_fn>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Transform
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TupleA, typename TupleB, typename Fn>
void test_transform()
{
    auto res = transform(TupleA{}, TupleB{}, Fn{});

    // clang-format off
    print("Input  : ");print(TupleA{});print(" , ");print(TupleB{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_transform_examples()
{
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
void test_find()
{
    auto res = find(T{}, X{});

    // clang-format off
    print("Input  : ");print(T{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_find_examples()
{
    test_find<tuple<_1, _4, _5, _2, _6>, _4>();
    test_find<tuple<_1, _4, _5, _2, _6>, _5>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Find if
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename F>
void test_find_if(T t, F &&f)
{
    auto res = find_if(T{}, f);

    // clang-format off
    print("Input  : ");print(T{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_find_if_examples()
{
    {
        print("Find the Value which greater than 5 and return index : \n");
        test_find_if(tuple<_1, _2, _5, _7>{}, [&](auto const &i)
                     { return greater{}(i, _5{}); });
    }
    {
        print("Find the Value which equal to 5 and return index : \n");
        test_find_if(tuple<_1, _2, _5, _7>{}, [&](auto const &i)
                     { return equal_to{}(i, _5{}); });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Compact Col Major
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Shape>
void test_compact_col_major()
{
    auto res = compact_col_major(Shape{});
    print("Input  : ");
    print(Shape{});
    print("\n");
    print("Output : ");
    print(res);
    print("\n");
}

void test_compact_col_major_examples()
{
    test_compact_col_major<Shape<_4>>();
    test_compact_col_major<Shape<_4, _2>>();
    test_compact_col_major<Shape<_4, _2, _8>>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Inverse seq
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Start, typename Shape, typename Stride>
void test_inverse_seq()
{
    auto res = detail::inverse_seq<Start>(Shape{}, Stride{}, seq<>{});

    // clang-format off
    print("Input  : ");print(Shape{});print(Stride{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_inverse_seq_examples()
{
    test_inverse_seq<1, tuple<_4>, tuple<_1>>();
    test_inverse_seq<1, tuple<_4, _4>, tuple<_1, _4>>();
    test_inverse_seq<1, tuple<_4, _4>, tuple<_1, _5>>();
    test_inverse_seq<1, tuple<_4, _5>, tuple<_1, _4>>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Right Inverse
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Layout>
void test_right_inverse(int ps)
{
    auto res = right_inverse(Layout{});

    // clang-format off
    print("Input  : ");print(Layout{});print("\n");
    print("Output : ");print(res);print("\n");
    // clang-format on
}

void test_right_inverse_examples(int ps)
{
    test_right_inverse<Layout<Shape<_4>, Stride<_1>>>(ps);
    test_right_inverse<Layout<Shape<_4, _4>, Stride<_1, _4>>>(ps);
    test_right_inverse<Layout<Shape<_4, _4>, Stride<_1, _5>>>(ps);
    test_right_inverse<Layout<Shape<_4, _5>, Stride<_1, _4>>>(ps);
    test_right_inverse<Layout<Shape<_4, _5>, Stride<_5, _1>>>(ps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Composition
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename B>
void test_composition(int ps)
{
    auto res = composition(A{}, B{});

    // clang-format off
    print("Input  : ");custom_print(A{},ps);print(" , ");custom_print(B{},ps);print("\n");
    print("Output : ");custom_print(res,ps);print("\n");
    // clang-format on
}

void test_composition_examples(int ps)
{

    test_composition<Layout<Shape<Int<20>, _2>, Stride<_16, _4>>,
                     Layout<Shape<_4, _5>, Stride<_1, _4>>>(ps);

    test_composition<Layout<Shape<_2, _32>, Stride<_32, _1>>,
                     Layout<Shape<Shape<_8, _4>, _8>, Stride<Stride<_8, _0>, _1>>>(ps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                Right Inverse of Ref then Compose to Src
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename LayoutRef, typename LayoutSrc>
void test_RIRCS(int ps)
{
    auto res = right_inverse(LayoutRef{}).compose(LayoutSrc{});

    // clang-format off
    print("Reference Layout : ");custom_print(LayoutRef{},ps);print("\n");
    print("Source    Layout : ");custom_print(LayoutSrc{},ps);print("\n");
    print("Result    Layout : ");custom_print(res,ps);print("\n");
    // clang-format on
}

void test_RIRCS_examples(int ps)
{
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

int main(int argc, char *argv[])
{
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
    test_RIRCS_examples(ps);
}