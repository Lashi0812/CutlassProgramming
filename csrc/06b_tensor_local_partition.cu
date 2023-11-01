#include <cute/tile.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>
#include <iostream>

using namespace cute;
using X = Underscore;

void test_product_each()
{
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
    std::cout << "Shape of thread Layout " << shape(tA) << std::endl;
    std::cout << "Product each  " << product_each(shape(tA)) << std::endl;
}

void test_get_flat_coord()
{
    // default is col Major order GenColMajor
    // auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
    // using the cols major order
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), GenRowMajor());
    std::cout << tA.get_flat_coord(0) << std::endl;
    std::cout << tA.get_flat_coord(1) << std::endl;
    std::cout << tA.get_flat_coord(2) << std::endl;
    std::cout << tA.get_flat_coord(3) << std::endl;
    std::cout << tA.get_flat_coord(25) << std::endl;
    std::cout << tA.get_flat_coord(50) << std::endl;
}

void test_local_partition()
{
    // get the local tiles
    auto matA_layout = make_layout(make_shape(Int<4096>{}, Int<4096>{}));
    auto matA_tensor = make_counting_tensor(matA_layout);
    auto blk_shape = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
    auto blk_coord = make_coord(1, 1, _);
    auto gA = local_tile(matA_tensor, blk_shape, blk_coord, Step<_1, X, _1>{});

    std::cout << "local Tile ";
    print(gA);
    std::cout << "\n";

    // partition the local tile
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));
    std::cout << local_partition(gA, tA, 0) << std::endl;
    // std::cout << local_partition(gA, tA, 1) << std::endl;
    // std::cout << local_partition(gA, tA, 10) << std::endl;
}

void test_local_partition_C()
{
    auto matC_layout = make_layout(make_shape(Int<4096>{}, Int<4096>{}));
    auto matC_tensor = make_counting_tensor(matC_layout);
    auto blk_shape = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
    auto blk_coord = make_coord(1, 1, _);
    auto gC = local_tile(matC_tensor, blk_shape, blk_coord, Step<_1, _1, X>{});
    
    
    auto block_layout = make_layout(make_shape(Int<128>{}, Int<8>{}));
    auto shared_tensor = make_counting_tensor(block_layout);
    auto thread_layout = make_layout(make_shape(Int<16>{}, Int<16>{}));
    auto tAsA = local_partition(shared_tensor, thread_layout, 0, Step<_1, X>{});
    auto tBsB = local_partition(shared_tensor, thread_layout, 0, Step<X, _1>{});
    auto tCgC = local_partition(gC, thread_layout, 0, Step<_1, _1>{});


    
    std::cout << tAsA << std::endl;
    std::cout << tBsB << std::endl;
    std::cout << tCgC << std::endl;
}

int main()
{
    // test_product_each();
    // test_get_flat_coord();
    // test_local_partition();
    test_local_partition_C();
}