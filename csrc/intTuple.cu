#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include <iostream>

int main()
{
    // cute::Shape<int,int> myShape;
    auto myShape = cute::make_shape(1, 2, 14, 34);
    std::cout << myShape << std::endl;
    std::cout << "get 3rd element " << cute::get<3>(myShape) << std::endl;
    std::cout << "Rank " << cute::rank(myShape) << std::endl;
    std::cout << "Depth " << cute::depth(myShape) << std::endl;
    std::cout << "Size " << cute::size(myShape) << std::endl;
}