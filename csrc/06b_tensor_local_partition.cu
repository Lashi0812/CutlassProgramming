#include <cute/tile.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/underscore.hpp>
#include <iostream>
#include <vector>

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

void test_local_partition_warpA()
{
    auto layout = make_layout(make_shape(_16{}, _8{}));
    auto tensor_a = make_counting_tensor(layout);

    auto tile = make_tile(make_layout(_2{}, _1{}),
                          make_layout(_2{}, _4{}));

    // std::cout << logical_divide(tensor_a, tile) << std::endl;
    // std::cout << zipped_divide(tensor_a, tile) << std::endl;
    // std::cout << local_partition(tensor_a, tile, 0) << std::endl;
    // std::cout << zipped_divide(tensor_a, tile)(_, 0) << std::endl;
    // std::cout << zipped_divide(tensor_a, tile)(_, 1) << std::endl;
    // std::cout << zipped_divide(tensor_a, tile)(_, 2) << std::endl;
    for (int i{0}; i < 32; ++i)
    {
        std::cout << "Thread : " << i << std::endl;
        std::cout << zipped_divide(tensor_a, tile)(_, i) << std::endl;
    }
}

void test_local_partition_warpB()
{
    auto layout = make_layout(make_shape(_8{}, _8{}));
    auto tensor_b = make_counting_tensor(layout);

    auto tile = make_tile(make_layout(_2{}, _1{}));

    // std::cout << logical_divide(tensor_b, tile) << std::endl;
    // std::cout << zipped_divide(tensor_b, tile) << std::endl;
    // std::cout << local_partition(tensor_b, tile, 0) << std::endl;
    // std::cout << zipped_divide(tensor_b, tile)(_, 0) << std::endl;
    // std::cout << zipped_divide(tensor_b, tile)(_, 1) << std::endl;
    // std::cout << zipped_divide(tensor_b, tile)(_, 2) << std::endl;

    for (int i{0}; i < 32; ++i)
    {
        std::cout << "Thread : " << i << std::endl;
        std::cout << zipped_divide(tensor_b, tile)(_, i) << std::endl;
    }
}

void test_local_partition_warpC()
{

    // auto tensorC = make_tensor(make_gmem_ptr(C),
    //                            make_layout(make_shape(_16{}, _8{})));
    // auto each_threadC = zipped_divide(tensorC,
    //                                   make_tile(make_layout(_2{}, _1{}),
    //                                             make_layout(_2{}, _4{})))(_, threadIdx.x);

    std::vector<int> vec_c(16*8,1) ;
    auto layout = make_layout(make_shape(_16{}, _8{}));
    // auto tensor_c = make_counting_tensor(layout);
    auto tensor_c = make_tensor(vec_c.data(),layout);


    auto tile = make_tile(make_layout(_2{}, _1{}),
                          make_layout(_2{}, _4{}));

    std::vector<int> dest_vec_c{0,1,2,3};
    // auto dest_tensorC = make_counting_tensor(make_layout(_4{}, _1{}));
    auto dest_tensorC = make_tensor(dest_vec_c.data(),make_layout(_4{}, _1{}));

    // std::cout << logical_divide(tensor_b, tile) << std::endl;
    // std::cout << zipped_divide(tensor_b, tile) << std::endl;
    // std::cout << local_partition(tensor_b, tile, 0) << std::endl;
    // std::cout << zipped_divide(tensor_b, tile)(_, 0) << std::endl;
    // std::cout << zipped_divide(tensor_b, tile)(_, 1) << std::endl;
    // std::cout << zipped_divide(tensor_b, tile)(_, 2) << std::endl;

    print_tensor(dest_tensorC);
    for (int i{0}; i < 32; ++i)
    {
        std::cout << "Thread : " << i << std::endl;
        auto each_thread = zipped_divide(tensor_c, tile)(_, i);
        std::cout << "Before Copy" << std::endl;
        std::cout << each_thread << std::endl;
        copy(dest_tensorC, each_thread);
        std::cout << "After Copy" << std::endl;
        std::cout << each_thread << std::endl;
    }
}

int main()
{
    // test_product_each();
    // test_get_flat_coord();
    // test_local_partition();
    // test_local_partition_C();
    // test_local_partition_warpA();
    // test_local_partition_warpB();
    test_local_partition_warpC();
}