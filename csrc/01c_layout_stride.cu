#include <cute/tensor.hpp>

using namespace cute;

void test_stride_zero()
{
    auto layout = make_layout(make_shape(4, 2), make_stride(1, 0));
    print_layout(layout);
}

void test_tile_stride_access()
{
    auto tensor = make_counting_tensor(make_layout(make_shape(2, 4), make_stride(1, 8)));
    auto tile = make_shape(2, 1);
    auto access = make_layout(make_shape(2, 4), make_stride(0, 1));
    print("tensor  : ");
    print_tensor(tensor);
    print("\n");
    for (int i{0}; i < size(access); ++i)
    {
        auto tiled = local_tile(tensor, tile, access(i));
        print("Thread%d  : ", i);
        print_tensor(tiled);
        print("\n");
    }
}

int main()
{
    // test_stride_zero();
    test_tile_stride_access();
}