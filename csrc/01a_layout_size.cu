#include <cute/layout.hpp>
#include <iostream>

using namespace cute;

int main()
{
    auto shape = make_shape(2,4,5,make_shape(6,2));
    std::cout << size(shape) << std::endl;
    std::cout << size<0>(shape) << std::endl;
    std::cout << size<1>(shape) << std::endl;
    std::cout << size<2>(shape) << std::endl;
    // get the nested size
    std::cout << size<3,1>(shape) << std::endl;
}