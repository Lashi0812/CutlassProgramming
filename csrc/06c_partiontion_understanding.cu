#include <cute/tensor.hpp>
#include <vector>

using namespace cute;

template <typename Layout, typename Tiler>
CUTE_HOST_DEVICE constexpr auto tiled_shape(Layout layout, Tiler tiler)
{
    return make_shape((size<0>(layout) / size<0>(tiler)), (size<1>(layout) / size<1>(tiler)));
}

template <typename Layout, typename Tiler>
CUTE_HOST_DEVICE constexpr int number_of_tiles(Layout layout, Tiler tiler)
{
    return size(tiled_shape(layout, tiler));
}

void test_blk_tiled_4x4_thread_tiled_1x2()
{
    auto tensor = make_counting_tensor(make_layout(make_shape(8, 8)));
    auto block_tile = make_shape(4, 4);
    auto thread_tile = make_shape(2, 1);

    for (int blk_idx{0}; blk_idx < number_of_tiles(tensor, block_tile); ++blk_idx)
    {
        auto tiled = local_tile(tensor, block_tile, blk_idx);
        print("Block %d : ", blk_idx);
        print_tensor(tiled);
        print("\n");
        for (int th_idx{0}; th_idx < number_of_tiles(tiled, thread_tile); ++th_idx)
        {
            auto partitioned = local_tile(tiled, thread_tile, th_idx);
            print("\tThread %d : ", th_idx);
            print_tensor(partitioned);
            print("\n");
        }
    }
}

void test_blk_tiled_4x4_thread_tiled_2x1()
{
    auto tensor = make_counting_tensor(make_layout(make_shape(8, 8)));
    auto block_tile = make_shape(4, 4);
    auto thread_tile = make_shape(1, 2);

    for (int blk_idx{0}; blk_idx < number_of_tiles(tensor, block_tile); ++blk_idx)
    {
        auto tiled = local_tile(tensor, block_tile, blk_idx);
        print("Block %d : ", blk_idx);
        print_tensor(tiled);
        print("\n");
        for (int th_idx{0}; th_idx < number_of_tiles(tiled, thread_tile); ++th_idx)
        {
            auto partitioned = local_tile(tiled, thread_tile, th_idx);
            print("\tThread %d : ", th_idx);
            print_tensor(partitioned);
            print("\n");
        }
    }
}

void test_blk_tiled_4x4_thread_tiled_2x1_col_access()
{
    auto tensor = make_counting_tensor(make_layout(make_shape(8, 8)));
    auto block_tile = make_shape(4, 4);
    auto thread_tile = make_shape(1, 2);

    for (int blk_idx{0}; blk_idx < number_of_tiles(tensor, block_tile); ++blk_idx)
    {
        auto tiled = local_tile(tensor, block_tile, blk_idx);
        print("Block %d : ", blk_idx);
        print_tensor(tiled);
        print("\n");
        for (int th_idx{0}; th_idx < number_of_tiles(tiled, thread_tile); ++th_idx)
        {
            auto partitioned = local_tile(tiled, thread_tile,
                                          make_layout(reverse(tiled_shape(tiled, thread_tile)), GenRowMajor())(th_idx));
            print("\tThread %d : ", th_idx);
            print_tensor(partitioned);
            print("\n");
        }
    }
}

template <typename BlockTile, typename ThreadLayout>
void test_blk_tiled_thread_part(BlockTile blk_tile, ThreadLayout th_layout)
{
    auto tensor = make_counting_tensor(make_layout(make_shape(8, 8)));
    auto block_tile = blk_tile;
    auto thread_layout = th_layout;

    for (int blk_idx{0}; blk_idx < number_of_tiles(tensor, block_tile); ++blk_idx)
    {
        auto tiled = local_tile(tensor, block_tile, blk_idx);
        print("Block %d : ", blk_idx);
        print_tensor(tiled);
        print("\n");
        for (int th_idx{0}; th_idx < size(thread_layout); ++th_idx)
        {
            auto partitioned = local_partition(tiled, thread_layout, th_idx);
            print("\tThread %d : ", th_idx);
            print_tensor(partitioned);
            print("\n");
        }
    }
}

template <typename BlockTile, typename ChunkTile, typename ThreadTile>
void test_block_tile_chunk_tile_thread_tile(BlockTile blk_tiler, ChunkTile chk_tiler, ThreadTile th_tiler)
{
    auto tensor = make_counting_tensor(make_layout(make_shape(8, 8)));
    for (int blk_idx{0}; blk_idx < number_of_tiles(tensor, blk_tiler); ++blk_idx)
    {
        auto blk_tiled = local_tile(tensor, blk_tiler, blk_idx);
        print("Block %d : ", blk_idx);
        print_tensor(blk_tiled);
        print("\n");
        for (int chk_idx{0}; chk_idx < number_of_tiles(blk_tiled, chk_tiler); ++chk_idx)
        {
            auto chk_tiled = local_tile(blk_tiled, chk_tiler, chk_idx);
            print("\tChunk %d : ", chk_idx);
            print_tensor(chk_tiled);
            print("\n");
            for (int th_idx{0}; th_idx < number_of_tiles(blk_tiled, th_tiler); ++th_idx)
            {
                auto th_tiled = local_tile(chk_tiled, th_tiler,
                                           make_layout(tiled_shape(blk_tiled, th_tiler), make_stride(0, 1))(th_idx));
                print("\tThread %d : ", th_idx);
                print_tensor(th_tiled);
                print("\n");
            }
        }
    }
}

template <typename TensorLayout, typename BlockTile, typename ChunkTile, typename ThreadLayout>
void test_block_tile_chunk_tile_thread_part(TensorLayout tensor_layout, BlockTile blk_tiler, ChunkTile chk_tiler, ThreadLayout th_layout)
{
    auto tensor = make_counting_tensor(tensor_layout);
    for (int blk_idx{0}; blk_idx < number_of_tiles(tensor, blk_tiler); ++blk_idx)
    {
        auto blk_tiled = local_tile(tensor, blk_tiler, blk_idx);
        print("Block %d : ", blk_idx);
        print_tensor(blk_tiled);
        print("\n");
        for (int chk_idx{0}; chk_idx < number_of_tiles(blk_tiled, chk_tiler); ++chk_idx)
        {
            auto chk_tiled = local_tile(blk_tiled, chk_tiler, chk_idx);
            print("\tChunk %d : ", chk_idx);
            print_tensor(chk_tiled);
            print("\n");
            for (int th_idx{0}; th_idx < size(th_layout); ++th_idx)
            {
                auto th_tiled = local_partition(chk_tiled, th_layout, th_idx, Step<_1, Underscore>{});
                print("\tThread %d : ", th_idx);
                print_tensor(th_tiled);
                print("\n");
            }
        }
    }
}

template <typename TensorLayout, typename BlockTile, typename BlockCoord,
          typename SMEMShape, typename ThreadLayout, typename ThreadStep>
void test_block_tile_chunk_tile_copy_smem_thread_partA(TensorLayout tensor_layout, BlockTile blk_tiler, BlockCoord blk_coord,
                                                       SMEMShape smem_shape, ThreadLayout th_layout, ThreadStep)
{
    std::vector<int> smem(size(smem_shape));

    auto tensor = make_counting_tensor(tensor_layout);
    auto smem_tensor_copy = make_tensor(smem.data(), make_layout(smem_shape));
    auto smem_tensor_part = make_tensor(smem.data(), make_layout(smem_shape, GenRowMajor()));

    auto blk_tiled = local_tile(tensor, blk_tiler, blk_coord);
    print("Block  : ");
    print_tensor(blk_tiled);
    print("\n");
    for (int chk_idx{0}; chk_idx < size<2>(blk_tiled); ++chk_idx)
    {
        auto chk_tiled = blk_tiled(_, _, chk_idx);
        copy(chk_tiled, smem_tensor_copy);
        print("\tChunk %d : ", chk_idx);
        print_tensor(chk_tiled);
        print("\n");
        print("\tSmem Data for Chunk %d : ", chk_idx);
        print_tensor(smem_tensor_copy);
        print_tensor(smem_tensor_part);
        print("\n");
        for (int th_idx{0}; th_idx < size(th_layout); ++th_idx)
        {
            auto th_tiled = local_partition(smem_tensor_part, th_layout, th_idx, ThreadStep{});
            print("\tThread %d : ", th_idx);
            print_tensor(th_tiled);
            print("\n");
        }
    }
}

template <typename TensorLayout, typename BlockTile, typename BlockCoord,
          typename SMEMLayout, typename ThreadLayout, typename ThreadStep>
void test_block_tile_chunk_tile_copy_smem_thread_partB(TensorLayout tensor_layout, BlockTile blk_tiler, BlockCoord blk_coord,
                                                       SMEMLayout smem_layout, ThreadLayout th_layout, ThreadStep)
{
    std::vector<int> smem(size(smem_layout));

    auto tensor = make_counting_tensor(tensor_layout);
    auto smem_tensor = make_tensor(smem.data(), smem_layout);

    auto blk_tiled = local_tile(tensor, blk_tiler, blk_coord);
    print("Block : ");
    print_tensor(blk_tiled);
    print("\n");
    for (int chk_idx{0}; chk_idx < size<2>(blk_tiled); ++chk_idx)
    {
        auto chk_tiled = blk_tiled(_, _, chk_idx);
        copy(chk_tiled, smem_tensor);
        print("\tChunk %d : ", chk_idx);
        print_tensor(chk_tiled);
        print("\n");
        print("\tSmem Data for Chunk %d : ", chk_idx);
        print_tensor(smem_tensor);
        print("\n");
        for (int th_idx{0}; th_idx < size(th_layout); ++th_idx)
        {
            auto th_tiled = local_partition(smem_tensor, th_layout, th_idx, ThreadStep{});
            print("\tThread %d : ", th_idx);
            print_tensor(th_tiled);
            print("\n");
        }
    }
}

int main()
{
    // test_blk_tiled_4x4_thread_tiled_1x2();
    // test_blk_tiled_4x4_thread_tiled_2x1();
    // test_blk_tiled_4x4_thread_tiled_2x1_col_access();
    // {
    //     // test_blk_tiled_4x4_thread_part_2x4;
    //     // mode1 x mode0  ==> 2x4
    //     // cute ==> mode0, mode1 ===> 4,2
    //     auto blk_tile = make_shape(4, 4);
    //     auto th_layout = make_layout(make_shape(4, 2));
    //     test_blk_tiled_thread_part(blk_tile, th_layout);
    // }
    // {
    //     // test_blk_tiled_4x4_thread_part_2x4_col_access;
    //     // mode1 x mode0  ==> 2x4
    //     // cute ==> mode0, mode1 ===> 4,2
    //     auto blk_tile = make_shape(4, 4);
    //     auto th_layout = make_layout(make_shape(4, 2), GenRowMajor());
    //     test_blk_tiled_thread_part(blk_tile, th_layout);
    // }
    // {
    //     // test_blk_tiled_4x4_thread_part_4x2;
    //     // mode1 x mode0  ==> 4x2
    //     // cute ==> mode0, mode1 ===> 2,4
    //     auto blk_tile = make_shape(4, 4);
    //     auto th_layout = make_layout(make_shape(2, 4));
    //     test_blk_tiled_thread_part(blk_tile, th_layout);
    // }
    // {
    //     auto blk_tiler = make_shape(4, 4);
    //     auto chk_tiler = make_shape(2, 4);
    //     auto th_tiler = make_shape(2, 1);
    //     test_block_tile_chunk_tile_thread_tile(blk_tiler, chk_tiler, th_tiler);
    // }

    // {
    //     // 8 partition
    //     // Think as the Row Wise as the old method
    //     // mode0 x mode1
    //     // chunk_tile ===> 4x2
    //     // cute_chunk_tile ==> 4,2
    //     auto tensor_layout = make_layout(make_shape(8, 8), GenRowMajor());
    //     auto blk_tiler = make_shape(4, 4);
    //     auto chk_tiler = make_shape(4, 2);
    //     auto th_layout = make_layout(make_shape(2, 4), GenRowMajor());
    //     test_block_tile_chunk_tile_thread_part(tensor_layout, blk_tiler, chk_tiler, th_layout);
    // }

    // {
    //     // 9 partition
    //     // Same as 8 but in cute way
    //     // mode1 x mode0
    //     // chunk_tile ===> 4x2
    //     // cute_chunk_tile ==> 2,4
    //     auto tensor_layout = make_layout(make_shape(8, 8));
    //     auto blk_tiler = make_shape(2, 4);
    //     auto blk_coord = make_coord(_, 0);
    //     auto smem_shape = make_shape(4, 2);
    //     auto th_layout = make_layout(make_shape(4, 2));
    //     test_block_tile_chunk_tile_copy_smem_thread_partA(tensor_layout, blk_tiler, blk_coord,
    //                                                       smem_shape, th_layout, Step<Underscore, _1>{});
    // }

    {
        // 10 partition
        auto tensor_layout = make_layout(make_shape(8, 8));
        auto blk_tiler = make_shape(4, 2);
        auto blk_coord = make_coord(0, _);
        auto smem_shape = make_shape(4, 2);
        auto th_layout = make_layout(make_shape(4, 2));
        test_block_tile_chunk_tile_copy_smem_thread_partB(tensor_layout, blk_tiler, blk_coord,
                                                          smem_shape, th_layout, Step<_1, Underscore>{});
    }
}