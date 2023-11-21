#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <vector>
#include <iostream>
#include <numeric>

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

template <typename SrcLayout, typename DstLayout>
void test_copy(SrcLayout src_layout, DstLayout dst_layout, bool manual = false)
{
    std::vector<int> src_vec(size(src_layout));
    std::vector<int> dst_vec(size(dst_layout));

    std::iota(src_vec.begin(), src_vec.end(), 0);
    auto src_tensor = make_tensor(src_vec.data(), src_layout);
    auto dst_tensor = make_tensor(dst_vec.data(), dst_layout);

    if (manual)
    {
        for (int i{0}; i < size(dst_layout); ++i)
        {
            dst_tensor(i) = src_tensor(i);
        }
    }
    else
    {
        copy(src_tensor, dst_tensor);
    }

    print("Source Tensor : ");
    print_tensor(src_tensor);
    print("\n");

    print("Destination Tensor : ");
    print_tensor(dst_tensor);
    print("\n");

    print("Destination Tensor : ");
    print_tensor(make_tensor(dst_vec.data(),make_layout(shape(dst_layout))));
    print("\n");
}

int main()
{
    // test_copy();
    // test_max_common_vec();
    // test_manual_copy();
    {
        // row row  copy
        auto src_layout = make_layout(make_shape(2, 4));
        auto dst_layout = make_layout(make_shape(4, 2));
        test_copy(src_layout, dst_layout);
    }
    {
        // row col copy
        auto src_layout = make_layout(make_shape(2, 4));
        auto dst_layout = make_layout(make_shape(4, 2),GenRowMajor());
        test_copy(src_layout, dst_layout);
    }
    {
        auto src_layout = make_layout(make_shape(2, 4));
        auto dst_layout = make_layout(make_shape(4, 2),GenRowMajor());
        test_copy(src_layout, dst_layout,true);
    }
}