#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>

#include <iostream>
using namespace cute;

int main()
{
    auto shape4x3 = make_shape(4, 3);

    auto colMajorStride = make_stride(1, 4);
    auto colMajorLayout = make_layout(shape4x3, colMajorStride);
    std::cout << "Column Major Layout " << std::endl;
    print_layout(colMajorLayout);

    auto rowMajorStride = make_stride(3, 1);
    auto rowMajorLayout = make_layout(shape4x3, rowMajorStride);
    std::cout << "Row Major Layout " << std::endl;
    print_layout(rowMajorLayout);

    auto shape8x8 = make_shape(make_shape(make_shape(2, 2), 2),
                               make_shape(make_shape(2, 2), 2));
    auto zOrderStride = make_stride(make_stride(make_stride(2, 8), 32),
                                    make_stride(make_stride(1, 4), 16));
    auto zOrderLayout = make_layout(shape8x8, zOrderStride);
    std::cout << "Z order Layout " << std::endl;
    print_layout(zOrderLayout);
}