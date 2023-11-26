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

int main()
{
    test_transform_leaf_examples();
}