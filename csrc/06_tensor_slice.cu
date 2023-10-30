#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <vector>
#include <iostream>
#include <cute/underscore.hpp>

using namespace cute;
int main()
{
    auto vec = std::vector{1, 5, 8, 11, 23, 12, 4, 86, 434};
    auto tensor = make_tensor(vec.data(), Layout<Shape<_3, _3>, Stride<_3, _1>>{});
    std::cout << tensor << std::endl;
    std::cout << "Row Slice " << std::endl;
    std::cout << tensor(make_coord(2, _));
    std::cout << "Col Slice " << std::endl;
    std::cout << tensor(make_coord(_, 2));
    
}