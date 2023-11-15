#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <vector>
#include <iostream>

using namespace cute;

void test_copy()
{
    auto layout = Layout<Shape<_2, _2>, Stride<_2, _1>>{};

    auto vec10 = std::vector{10, 11, 12, 13};
    auto ten10 = make_tensor(vec10.data(), layout);
    print_tensor(ten10);

    auto vec20 = std::vector{20, 21, 22, 23};
    auto ten20 = make_tensor(vec20.data(), layout);
    print_tensor(ten20);

    copy(ten10(_, 1), ten20);
    print_tensor(ten10);
    print_tensor(ten20);

    copy(ten10(1, _), ten20);
    print_tensor(ten10);
    print_tensor(ten20);

    copy(ten10, ten20);
    print_tensor(ten10);
    print_tensor(ten20);
}

void test_max_common_vec()
{
    auto layout = Layout<Shape<_2, _2>, Stride<_2, _1>>{};

    auto vec10 = std::vector{10, 11, 12, 13};
    auto ten10 = make_tensor(vec10.data(), layout);

    auto vec20 = std::vector{20, 21, 22, 23};
    auto ten20 = make_tensor(vec20.data(), layout);

    max_common_vector(ten10, ten20);
    copy(ten10, ten20);
}

void test_manual_copy()
{
    auto layout = Layout<Shape<_2, _2>, Stride<_2, _1>>{};

    auto vec10 = std::vector{10, 11, 12, 13};
    auto ten10 = make_tensor(vec10.data(), layout);

    auto vec20 = std::vector{20, 21, 22, 23};
    auto ten20 = make_tensor(vec20.data(), layout);

    print("Before Copy: \n");
    print_tensor(ten10);
    print_tensor(ten20);

    for (int i{0}; i < size(ten10); ++i)
    {
        ten10(i) = ten20(i);
    }

    print("After Copy: \n");
    print_tensor(ten10);
    print_tensor(ten20);
}

int main()
{
    // test_copy();
    // test_max_common_vec();
    test_manual_copy();
}