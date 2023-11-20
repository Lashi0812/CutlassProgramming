#include <cute/tensor.hpp>

using namespace cute;

void test_partitionC_for_smem1D(int matMode1, int matMode2,
                                int blkMode1, int blkMode2,
                                int thMode1, int thMode2)
{
    auto matLayout = make_layout(make_shape(matMode1, matMode2));
    auto threadLayout = make_layout(make_shape(thMode1, thMode2));
    auto tensorC = make_counting_tensor(matLayout);

    auto block_partition = local_tile(tensorC, make_shape(blkMode1, blkMode2), make_coord(0, 1));
    print("Partitioned C Tensor : ");
    print_tensor(block_partition);
    print("\n");
    for (int i{0}; i < size(threadLayout); ++i)
    {
        auto thread_partition = local_partition(block_partition, threadLayout, i);
        print("C Data for thread%d : \n", i);
        print_tensor(thread_partition);
        print("\n");
    }
}

void test_partitionA_for_smem1D(int matMode1, int matMode2,
                                int blkMode1, int blkMode2,
                                int thMode1, int thMode2)
{
    auto matLayout = make_layout(make_shape(matMode1, matMode2), GenRowMajor());
    auto threadLayout = make_layout(make_shape(thMode1, thMode2));
    auto tensorA = make_counting_tensor(matLayout);

    auto block_partition = local_tile(tensorA, make_shape(blkMode1, blkMode2), make_coord(1, _));
    print("Partitioned A Tensor : ");
    print_tensor(block_partition);
    print("\n");

    auto smem_data = block_partition(_, _, 0);
    print("Smem data for A : ");
    print_tensor(smem_data);
    print("\n");
    for (int i{0}; i < size(threadLayout); ++i)
    {
        auto thread_partition = local_partition(smem_data, threadLayout, i, Step<Underscore, _1
        >{});
        print("A Data for thread%d : \n", i);
        print_tensor(thread_partition);
        print("\n");
    }
}

void test_partitionB_for_smem1D(int matMode1, int matMode2,
                                int blkMode1, int blkMode2,
                                int thMode1, int thMode2)
{
    auto matLayout = make_layout(make_shape(matMode1, matMode2));
    auto threadLayout = make_layout(make_shape(thMode1, thMode2));
    auto tensor = make_counting_tensor(matLayout);

    auto block_partition = local_tile(tensor, make_shape(blkMode1, blkMode2), make_coord(0, _));
    print("Partitioned B Tensor : ");
    print_tensor(block_partition);
    print("\n");

    auto smem_data = block_partition(_, _, 0);
    print("Smem data for B : ");
    print_tensor(smem_data);
    print("\n");
    for (int i{0}; i < size(threadLayout); ++i)
    {
        auto thread_partition = local_partition(smem_data, threadLayout, i, Step<_1, Underscore>{});
        print("B Data for thread%d : \n", i);
        print_tensor(thread_partition);
        print("\n");
    }
}

int main()
{
    {
        auto M{8}, N{8}, BM{4}, BN{4}, TM{2}, TN{4};
        // test_partitionC_for_smem1D(M, N,
        //                            BM, BN,
        //                            TM, TN);

        test_partitionC_for_smem1D(N, M,
                                   BN, BM,
                                   TN, TM);
    }

    {
        auto M{8}, K{8}, BM{4}, BK{2}, TM{2}, TN{4};
        test_partitionA_for_smem1D(M, K,
                                   BM, BK,
                                   TN, TM);
    }

    {
        auto N{8}, K{8}, BN{4}, BK{2}, TM{2}, TN{4};
        test_partitionB_for_smem1D(N, K,
                                   BN, BK,
                                   TN, TM);
    }
}