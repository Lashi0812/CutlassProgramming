#include <cute/layout.hpp>
#include <iostream>

using namespace cute;

int main()
{
    // auto tiles = make_layout(make_shape(2,2),make_stride(1,2));
    // auto matrixOfTiles = make_layout(make_shape(3,4),make_stride(4,1));
    auto tiles = Layout<Shape<_2,_2>,Stride<_1,_2>>{};
    auto matrixOfTiles = Layout<Shape<_3,_4>,Stride<_4,_1>>{};
    std::cout << "Blocked Product " << std::endl; 
    print_layout(blocked_product(tiles,matrixOfTiles));
    std::cout << "Raked Product " << std::endl;
    print_layout(raked_product(tiles,matrixOfTiles));
    std::cout << "zipped Product " << std::endl;
    print_layout(zipped_product(tiles,matrixOfTiles));
    std::cout << "Logical Product " << std::endl;
    print_layout(logical_product(tiles,matrixOfTiles));

}