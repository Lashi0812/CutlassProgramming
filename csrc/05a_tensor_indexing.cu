#include <cute/tensor.hpp>
#include <vector>

using namespace cute;

void test_tiled_indexing()
{
    auto tile = make_shape(_2{}, _1{});

    auto tensor = make_counting_tensor(make_layout(make_shape(_4{}, _4{}), make_stride(_1{}, _8{})));
    print("Original tensor");
    print_tensor(tensor);
    print("\n");

    auto log_div = logical_divide(tensor, tile);
    print("Logical Tiled Tensor : ");
    print_tensor(log_div);
    print("\n");

    auto zip_div = zipped_divide(tensor, tile);
    print("Zipped Tiled Tensor : ");
    print_tensor(zip_div);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Idx %d : Data %d \n", i,tensor(i));
        print("\t1D Coord : ");
        print(tensor.get_1d_coord(layout(tensor)(i)));
        print(" , ");
        print(log_div.get_1d_coord(layout(tensor)(i)));
        print(" , ");
        print(zip_div.get_1d_coord(layout(tensor)(i)));
        print("\n");

        print("\tFlat Coord : ");
        print(tensor.get_flat_coord(layout(tensor)(i)));
        print(" , ");
        print(log_div.get_flat_coord(layout(tensor)(i)));
        print(" , ");
        print(zip_div.get_flat_coord(layout(tensor)(i)));
        print("\n");

        print("\tHier Coord : ");
        print(tensor.get_hier_coord(layout(tensor)(i)));
        print(" , ");
        print(log_div.get_hier_coord(layout(tensor)(i)));
        print(" , ");
        print(zip_div.get_hier_coord(layout(tensor)(i)));
        print("\n");
    }
}

void test_vec_tiled_indexing()
{
    auto tile = make_shape(_2{}, _1{});
    std::vector vec{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};

    auto tensor = make_tensor(vec.data(), make_layout(make_shape(_4{}, _4{})));
    print("Original tensor");
    print_tensor(tensor);
    print("\n");

    auto log_div = logical_divide(tensor, tile);
    print("Logical Tiled Tensor : ");
    print_tensor(log_div);
    print("\n");

    auto zip_div = zipped_divide(tensor, tile);
    print("Zipped Tiled Tensor : ");
    print_tensor(zip_div);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Idx %d : \n", i);
        print("\t1D Coord : ");
        print(tensor.get_1d_coord(i));
        print(" , ");
        print(log_div.get_1d_coord(i));
        print(" , ");
        print(zip_div.get_1d_coord(i));
        print("\n");

        print("\tFlat Coord : ");
        print(tensor.get_flat_coord(i));
        print(" , ");
        print(log_div.get_flat_coord(i));
        print(" , ");
        print(zip_div.get_flat_coord(i));
        print("\n");

        print("\tHier Coord : ");
        print(tensor.get_hier_coord(i));
        print(" , ");
        print(log_div.get_hier_coord(i));
        print(" , ");
        print(zip_div.get_hier_coord(i));
        print("\n");
    }
}
int main()
{
    test_tiled_indexing();
    // test_vec_tiled_indexing();
}