#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>

#include <iostream>

int main()
{
    auto shape4x3 = cute::make_shape(4, 3);

    auto colMajorStride = cute::make_stride(1, 4);
    auto colMajorLayout = cute::make_layout(shape4x3, colMajorStride);
    std::cout << "Column Major Layout " << std::endl;
    cute::print_layout(colMajorLayout);

    auto rowMajorStride = cute::make_stride(3,1);
    auto rowMajorLayout = cute::make_layout(shape4x3,rowMajorStride);
    std::cout << "Row Major Layout " << std::endl;
    cute::print_layout(rowMajorLayout);
}