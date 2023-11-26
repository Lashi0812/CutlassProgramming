#include <cute/layout.hpp>

using namespace cute;

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
    test_find<tuple<_1,_4,_5,_2,_6>,_4>();
    test_find<tuple<_1,_4,_5,_2,_6>,_5>();
}

int main()
{
    // test_transform_leaf_examples();
    // test_transform_examples();
    test_find_examples();
}