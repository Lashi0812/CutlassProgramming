#include <cute/tensor.hpp>
#include <cute/tile.hpp>
#include <cute/underscore.hpp>
#include <iostream>

using namespace cute;
using X = Underscore;

void test()
{
    auto tensor = make_counting_tensor(Layout<Shape<_8, _8>, Stride<_1, _8>>{});
    print_tensor(tensor);
    auto tile = make_shape(2, 2, 2);
    auto coo = make_coord(1, 2, _);

    std::cout << dice(Step<_1, X, _1>{}, tile) << std::endl;
    std::cout << dice(Step<_1, X, _1>{}, coo) << std::endl;

    // std::cout << tensor(coo) << std::endl;
    auto some = local_tile(tensor, tile, coo, Step<_1, X, _1>{});
    // std::cout << local_tile(tensor, tile, coo, Step<_1, X, _1>{}) << std::endl;
    // std::cout << local_tile(tensor, tile, coo, Step<X, _1, _1>{}) << std::endl;
    // std::cout << local_tile(tensor, tile, coo, Step<_1, _1, X>{}) << std::endl;

    auto some2 = local_partition(some, Layout<Shape<_4, _4>>{}, 1);
    std::cout << some << std::endl;
    std::cout << some2 << std::endl;
}

void test_repeat()
{
    auto raked_layout = Layout<Shape<Shape<_3, _2>, Shape<_4, _2>>,
                               Stride<Stride<_16, _1>, Stride<_4, _2>>>{};

    print_layout(raked_layout);
    auto subtile = make_tile(Layout<_2, _3>{},
                             Layout<_2, _4>{});
    std::cout << subtile << std::endl;
    constexpr int R0 = decltype(rank(subtile))::value;
    constexpr int R1 = decltype(rank(raked_layout))::value;

    std::cout << "Rank " << R0 << " , " << R1 << std::endl;

    std::cout << repeat<R0>(_) << std::endl;
    std::cout << append<5>(make_coord(2, 3), _) << std::endl;
}

void test_zipped()
{
    auto raked_layout = Layout<Shape<Shape<_3, _2>, Shape<_4, _2>>,
                               Stride<Stride<_16, _1>, Stride<_4, _2>>>{};

    print_layout(raked_layout);
    auto subtile = make_tile(Layout<_2, _3>{},
                             Layout<_2, _4>{});

    auto zipped = zipped_divide(raked_layout, subtile);
    print_layout(zipped);

    std::cout << zipped(2, 3) << std::endl;
}

void test_local_tile()
{
    auto matA_layout = make_layout(make_shape(Int<4096>{}, Int<4096>{}),
                                   make_stride(Int<1>{}, Int<4096>{}));
    auto matA = make_counting_tensor(matA_layout);
    auto blk_shape = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
    auto coord = make_coord(1, 1, _);

    auto gA = local_tile(matA, blk_shape, coord, Step<_1, X, _1>{});
    auto gATest = local_tile(matA, make_shape(Int<128>{}, Int<8>{}), make_coord(1, 1));

    std::cout << "MatA ";
    print(matA);
    std::cout << "\n";
    std::cout << "A tile ";
    print(gA);
    std::cout << "\n";
    std::cout << "A tile Test ";
    print(gATest);
    std::cout << "\n";
    // std::cout << gA(0, 0, _) << std::endl;
}

void test_local_tile_vs_manual()
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

    print("Manual All thread in all blocks in the 0th Col : ");
    print_tensor(divided(_, make_coord(_, 0)));
    print("\n");

    print("Local tile All thread in all blocks in the 0th Col : ");
    print_tensor(local_tile(tensor, shape, make_coord(_, 0)));
    print("\n");
}

void test_local()
{
    std::vector<int> vec{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33};
    auto tensorA = make_tensor(vec.data(), make_layout(make_shape(_4{}, _6{}), GenRowMajor()));
    auto tensorB = make_tensor(vec.data(), make_layout(make_shape(_6{}, _4{})));

    print_tensor(tensorA);
    auto tiledA = local_tile(tensorA, make_shape(_2{}, _3{}), make_coord(0, _));
    print_tensor(tiledA);

    print_tensor(tensorB);
    auto tiledB = local_tile(tensorB, make_shape(_3{}, _2{}), make_coord(_, 0));
    print_tensor(tiledB);
}

void test_local_tile_shared_to_rmem()
{
    auto tensor = make_counting_tensor(make_layout(make_shape(_2{}, _2{})));
    print_tensor(tensor);
    auto tiled = local_tile(tensor, make_shape(_1{}, _1{}), make_coord(_, 0));
    print_tensor(tiled);

    auto part = local_partition(tiled, make_layout(make_shape(_1{}, _1{})), 0);
    print_tensor(part);
}

void test_get_all_block()
{
    auto tensor = make_counting_tensor(make_layout(make_shape(4, 4)));
    auto tiled = local_tile(tensor, make_shape(2, 2), make_coord(_, _));

    print("Tensor : ");
    print_tensor(tensor);
    print("\n");
    print("All Blocks in both axis: ");
    print_tensor(tiled);
    print("\n");
    print("All Block for 0th row along axis1 : ");
    print_tensor(tiled(_, _, 0, _));
    print("\n");
    print("All Block for 1th col along axis0 : ");
    print_tensor(tiled(_, _, _, 1));
    print("\n");
}

void test_1d_local_tileB()
{
    auto layoutC = make_layout(make_shape(_4{}, _4{}));
    auto tensor = make_counting_tensor(make_layout(make_shape(_2{}, _4{})));
    auto tile = make_shape(_2{}, _1{});

    auto logicalC = zipped_divide(layoutC, tile);
    print(logicalC);

    print_tensor(tensor);

    for (int i{0}; i < size(layoutC); ++i)
    {
        auto tiled = local_tile(tensor, tile, get<1>(logicalC.get_hier_coord(i)), Step<_0, _1>{});
        print("For Thread %d :", i);
        print_tensor(tiled);
        print("\n");
    }
}

void test_1d_local_tileA()
{
    auto layoutC = make_layout(make_shape(_4{}, _4{}));
    auto tensor = make_counting_tensor(make_layout(make_shape(_4{}, _2{})));
    auto tile = make_shape(_2{}, _1{});

    auto logicalC = zipped_divide(layoutC, tile);
    print(logicalC);

    print_tensor(tensor);
    for (int i{0}; i < size(layoutC); ++i)
    {
        print(logicalC.get_hier_coord(i));
        print("\n");
        auto tiled = local_tile(tensor, tile, get<1>(logicalC.get_hier_coord(i)), Step<_1, Underscore>{});
        print("For Thread %d :", i);
        print_tensor(tiled);
        print("\n");
    }
}

int main()
{
    // test();
    // test_repeat();
    // test_zipped();
    // test_local_tile();
    // test_local_tile_vs_manual();
    // test_local();
    // test_local_tile_shared_to_rmem();
    // test_get_all_block();
    test_1d_local_tileB();
    test_1d_local_tileA();
}