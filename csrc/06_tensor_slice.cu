#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/tile.hpp>
#include <vector>
#include <iostream>
#include <cute/underscore.hpp>

using namespace cute;

void test_tensor_slice()
{

    auto vec = std::vector{1, 5, 8, 11, 23, 12, 4, 86, 434};
    auto tensor = make_tensor(vec.data(), Layout<Shape<_3, _3>, Stride<_3, _1>>{});
    std::cout << tensor << std::endl;
    std::cout << "Row Slice " << std::endl;
    std::cout << tensor(make_coord(2, _));
    std::cout << "Col Slice " << std::endl;
    std::cout << tensor(make_coord(_, 2));
}

void test_block_slice_in_block_product()
{
    auto threadBlock = make_layout(make_shape(_2{}, _3{}));
    auto blockGrid = make_layout(make_shape(_3{}, _2{}));
    auto grid = blocked_product(threadBlock, blockGrid);

    print_layout(grid);
    print_layout(grid(make_coord(make_coord(0, 0), make_coord(_, _))));
    print_layout(grid(make_coord(make_coord(_, _), make_coord(0, 0))));
}

void test_block_slice_in_zipped_product()
{
    auto threadBlock = make_layout(make_shape(_2{}, _3{}));
    auto blockGrid = make_layout(make_shape(_2{}, _2{}));
    auto grid = zipped_product(threadBlock, blockGrid);

    print_layout(grid);
    // all the thread in block0
    print_layout(grid(make_coord(make_coord(_, _), make_coord(0, 0))));
    // all the thread in block in 0th row
    print_layout(grid(_, make_coord(0, _)));
    // all the thread0 in all the block
    print_layout(grid(make_coord(make_coord(0, 0), make_coord(_, _))));
    // all the thread in block in 0th col
    print_layout(grid(_, make_coord(_, 0)));
}

void test_logical_divide()
{
    auto matC = make_layout(make_shape(_4{}, _4{}));
    auto tile = make_tile(make_layout(make_shape(_2{})),
                          make_layout(make_shape(_2{})));
    auto divide = logical_divide(matC, tile);
    print_layout(divide);

    std::cout << divide.get_flat_coord(1) << std::endl;
    std::cout << divide.get_hier_coord(1) << std::endl;
    std::cout << divide.get_hier_coord(2) << std::endl;
    std::cout << divide.get_hier_coord(3) << std::endl;
    std::cout << divide.get_hier_coord(4) << std::endl;
    std::cout << divide.get_hier_coord(8) << std::endl;

    std::cout << divide(make_coord(make_coord(make_coord(_), _), make_coord(make_coord(0), 1))) << std::endl;
    print_layout(divide(make_coord(make_coord(make_coord(_), 0), make_coord(make_coord(_), 0))));
    print_layout(divide(make_coord(make_coord(make_coord(0), _), make_coord(make_coord(0), _))));
    print_layout(divide(make_coord(make_coord(make_coord(1), _), make_coord(make_coord(0), _))));
}

void test_zipped_divide()
{
    auto matC_layout = make_layout(make_shape(_4{}, _4{}));
    auto matC = make_counting_tensor(matC_layout);
    auto thread_layout = make_layout(make_shape(_2{}, _2{}));
    auto divide_layout = zipped_divide(matC, thread_layout);

    auto thread_tile = make_tile(make_layout(_2{}), make_layout(_2{}));
    auto divide_tile = zipped_divide(matC, thread_tile);

    print("Zipped layout : ");
    print_tensor(divide_layout);
    print("\n");
    
    print("Zipped tiled layout : ");
    print_tensor(divide_tile);
    print("\n");

    print("(0,1)Thread in (1,1) block : ");
    print(divide_tile(make_coord(0, 1), make_coord(1, 1)));
    print("\n");

    print("All threads in block0 : ");
    print_tensor(divide_tile(make_coord(_, _), make_coord(0, 0)));
    print("\n");

    print("All thread0 in all blocks : ");
    print_tensor(divide_tile(make_coord(0, 0), make_coord(_, _)));
    print("\n");

    print("All thread1 in all blocks : ");
    print_tensor(divide_tile(make_coord(0, 1), make_coord(_, _)));
    print("\n");

    print("All thread in all blocks of the 0th col : ");
    print_tensor(divide_tile(_, make_coord(_, 0)));
    print("\n");

    print("All thread in all blocks of the 1th col : ");
    print_tensor(divide_tile(_, make_coord(_, 1)));
    print("\n");

    print("All thread in all blocks of the 0th row : ");
    print_tensor(divide_tile(_, make_coord(0, _)));
    print("\n");

    print("All thread in all blocks of the 1th row : ");
    print_tensor(divide_tile(_, make_coord(1, _)));
    print("\n");
}

int main()
{
    // test_tensor_slice();
    // test_block_slice_in_block_product();
    // test_block_slice_in_zipped_product();
    // test_logical_divide();
    test_zipped_divide();
}