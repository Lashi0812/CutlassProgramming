#include <cute/tensor.hpp>
#include <cute/underscore.hpp>
#include <iostream>

using namespace cute;
using X = Underscore;

void test_logical_div_round_up()
{
    auto vec_layout = make_layout(make_shape(Int<100>{}));
    auto tile_layout = make_layout(_16{}, _1{});
    auto final_layout = logical_divide(vec_layout, tile_layout);
    print_layout(final_layout);

    std::cout << "Size of the Original Layout " << decltype(size(vec_layout))::value << std::endl;
    std::cout << "Size of the Final  Layout " << decltype(size(final_layout))::value << std::endl;
}

void test_solve_through_predicate()
{
    auto vec_layout = make_layout(make_shape(Int<100>{}));
    auto tile_layout = make_layout(_16{}, _1{});
    auto final_layout = logical_divide(vec_layout, tile_layout);

    auto pred = make_tensor<bool>(shape(final_layout));
    for (int i{0}; i < size(pred); ++i)
        pred(i) = final_layout(i) < size(vec_layout);

    // print_tensor(pred);
    std::cout << pred << std::endl;
}

void test_stride_zero()
{
    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_1{}, _0{}));
    print_layout(layout);
}

void test_local_partition()
{
    auto blockA_layout = make_layout(make_shape(Int<32>{}, Int<3>{}));
    auto blockA_tensor = make_counting_tensor(blockA_layout);
    auto threadA_layout = make_layout(make_shape(Int<8>{}, Int<2>{}));

    print_layout(blockA_layout);
    print_layout(threadA_layout);

    for (int tid{0}; tid < size(threadA_layout); ++tid)
    {
        std::cout << " thread id : " << tid << std::endl;
        auto each_thread = local_partition(blockA_tensor, threadA_layout, tid);
        
        std::cout << each_thread << std::endl;

        auto predA_tensor = make_tensor<bool>(make_shape(size<0>(each_thread), size<1>(each_thread)),
                                              make_stride(Int<0>{}, Int<1>{}));

        auto cA = make_identity_tensor(make_shape(size<0>(blockA_layout), size<1>(blockA_layout)));
        auto tAcA = local_partition(cA, threadA_layout, tid);

        // std::cout << cA << std::endl;
        // std::cout << tAcA << std::endl;
        // // std::cout << get<1>(shape(cA)) << std::endl;
        // // std::cout << get<1>(tAcA(0, 1)) << std::endl;

        for (int i{0}; i < size<1>(predA_tensor); ++i)
        {
            predA_tensor(0, i) = get<1>(tAcA(0, i)) < get<1>(shape(cA));
        }

        std::cout << predA_tensor << std::endl;
    }
}

// void test_tensor_pred()
// {
//     //  shapes
//     auto M = Int<4096>{};
//     auto N = Int<4096>{};
//     auto K = Int<4096>{};

//     auto BM = Int<128>{};
//     auto BN = Int<128>{};
//     auto BK = Int<8>{};

//     auto TM = Int<32>{};
//     auto TN = Int<32>{};
//     auto TK = Int<8>{};

//     auto TCM = Int<16>{};
//     auto TCN = Int<16>{};

//     // block layout
//     auto blockA = make_counting_tensor(make_layout(make_shape(BM, BK)));
//     auto blockB = make_counting_tensor(make_layout(make_shape(BN, BK)));
//     auto blockC = make_counting_tensor(make_layout(make_shape(BM, BN)));

//     // thread layout
//     auto threadA = make_counting_tensor(make_layout(make_shape(TM, TK)));
//     auto threadB = make_counting_tensor(make_layout(make_shape(TN, TK)));
//     auto threadC = make_counting_tensor(make_layout(make_shape(TCM, TCN)));

//     // represent full tensor
//     auto matA = make_counting_tensor(make_layout(make_shape(M, K),
//                                                  make_stride(Int<1>{}, K)));
//     auto matB = make_counting_tensor(make_layout(make_shape(N, K),
//                                                  make_stride(Int<1>{}, K)));
//     auto matC = make_counting_tensor(make_layout(make_shape(M, N),
//                                                  make_stride(Int<1>{}, N)));

//     // block shape and blk coord
//     auto blk_shape = make_shape(BM, BN, BK);
//     auto blk_coord = make_coord(_1{}, _1{}, _);

//     // tile the mat
//     auto gA = local_tile(matA, blk_shape, blk_coord, Step<_1, X, _1>{});
//     auto gB = local_tile(matB, blk_shape, blk_coord, Step<X, _1, _1>{});
//     auto gC = local_tile(matC, blk_shape, blk_coord, Step<_1, _1, X>{});

//     // local partition
//     auto tAgA = local_partition(gA, threadA, 0);
//     auto tBgB = local_partition(gB, threadA, 0);

//     // pred tensor
//     auto tApA = make_tensor<bool>(make_shape(size<0>(tAgA), size<1>(tAgA)),
//                                   make_stride(Int<1>{}, Int<0>{}));
//     auto tBpB = make_tensor<bool>(make_shape(size<0>(tBgB), size<1>(tBgB)),
//                                   make_stride(Int<1>{}, Int<0>{}));
// }

int main()
{
    // test_logical_div_round_up();
    // test_solve_through_predicate();
    // test_stride_zero();
    test_local_partition();
}
