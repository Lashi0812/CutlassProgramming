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

    std::vector<int> vec_c(16 * 8, 1);
    auto layout = make_layout(make_shape(_16{}, _8{}));
    // auto tensor_c = make_counting_tensor(layout);
    auto tensor_c = make_tensor(vec_c.data(), layout);

    auto tile = make_tile(make_layout(_2{}, _1{}),
                          make_layout(_2{}, _4{}));

    std::vector<int> dest_vec_c{0, 1, 2, 3};
    // auto dest_tensorC = make_counting_tensor(make_layout(_4{}, _1{}));
    auto dest_tensorC = make_tensor(dest_vec_c.data(), make_layout(_4{}, _1{}));

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

void test_local_partition_thread()
{
    auto M = _4{};
    auto N = _4{};

    auto threadLayout = make_tile(make_layout(_2{}), make_layout(_2{}));

    auto matLayout = make_layout(make_shape(M, N));
    auto tensorC = make_counting_tensor(matLayout);
    auto tensorA = make_counting_tensor(matLayout);
    auto tensorB = make_counting_tensor(matLayout);

    print("Tensor Of C : ");
    print_tensor(tensorC);
    print("\n");

    // considering the thread (0,1) in block (1,1)
    auto threadX = _1{};
    auto threadY = _0{};
    auto blockX = _1{};
    auto blockY = _1{};

    // considering the thread (0,1) in block (1,1) will do work
    auto threadC = zipped_divide(tensorC, threadLayout)(make_coord(threadY, threadX), make_coord(blockY, blockX));
    auto threadC_coord = zipped_divide(tensorC.layout(), threadLayout)(make_coord(threadY, threadX), make_coord(blockY, blockX));

    print("thread (0,1) in block (0,1) will do work for : %d \n", decltype(threadC_coord)::value);

    // gather all the element from A that need for calculate c
    auto neededA = tensorA(get<0>(tensorA.get_flat_coord(threadC_coord)), _);
    print("Need A element : ");
    print_tensor(neededA);
    print("\n");

    auto neededB = tensorB(_, get<1>(tensorB.get_flat_coord(threadC_coord)));
    print("Need B element : ");
    print_tensor(neededB);
    print("\n");
}

void test_single_tensor()
{
    auto tensor = make_counting_tensor(make_layout(make_shape(2, 2)));
    auto const &[sliced_layout, offset] = slice_and_offset(make_coord(0, 1), tensor.layout());
    print(offset);
    print("\n");
    print(sliced_layout);
    print("\n");

    // print(tensor(0,0));
    // print(tensor[0]);
    // print("\n");
    // print_tensor(tensor);
}

void test_local_partition_vs_manual()
{
    auto tensor_layout = make_layout(make_shape(_8{}, _4{}));
    auto shape = make_shape(_2{}, _2{});

    auto tensor = make_counting_tensor(tensor_layout);
    print("Tensor : ");
    print_tensor(tensor);
    print("\n");

    auto divided = zipped_divide(tensor, shape);
    print("Zipped Divided : ");
    print_tensor(divided);
    print("\n");

    auto manual_tile = divided(_, make_coord(_, 0));
    print("Manual All thread in all blocks in the 0th Col : ");
    print_tensor(manual_tile);
    print("\n");

    auto manual_part = manual_tile(make_coord(0,0),_);
    print("Element that Need by the thread0 : ");
    print_tensor(manual_part);
    print("\n");

    auto local_tileA = local_tile(tensor,shape,make_coord(_,0));
    auto local_partA = local_partition(local_tileA,shape,0);

    print("Using the fn Element that Need by the thread0 : ");
    print_tensor(local_partA); 
    print("\n");
}


int main()
{
    // test_product_each();
    // test_get_flat_coord();
    // test_local_partition();
    // test_local_partition_C();
    // test_local_partition_warpA();
    // test_local_partition_warpB();
    // test_local_partition_warpC();
    // test_local_partition_thread();
    // test_single_tensor();
    test_local_partition_vs_manual();
}
