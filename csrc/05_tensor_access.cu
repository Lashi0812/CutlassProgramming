#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <vector>
#include <iostream>

using namespace cute;

void test_access()
{
    auto vec = std::vector{1, 5, 8, 11};
    auto tensor = make_tensor(vec.data(), Layout<Shape<_2, _2>, Stride<_1, _2>>{});
    print_tensor(tensor);
    std::cout << "Array Access " << std::endl;
    // python way of access
    std::cout << tensor[1] << std::endl;

    // specify the linear index
    std::cout << tensor(0) << std::endl;
    std::cout << tensor(1) << std::endl;
    std::cout << tensor(2) << std::endl;
    std::cout << tensor(3) << std::endl;

    // specify row and col
    std::cout << tensor(0, 0) << std::endl;
    std::cout << tensor(0, 1) << std::endl;
    std::cout << tensor(1, 0) << std::endl;
    std::cout << tensor(1, 1) << std::endl;
}

template <class Engine, class Layout>
void print_char_tensor(Tensor<Engine, Layout> const &tensor)
{
    print(tensor);
    print(":\n");
    for (int m = 0; m < size<0>(tensor); ++m)
    {
        for (int n = 0; n < size<1>(tensor); ++n)
        {
            printf("%c   ", tensor(m, n));
        }
        printf("\n");
    }
}

void test_data_row_cute_row_not_transposed()
{
    std::vector<char> row_vec{'a', 'b', 'c', 'd', 'e', 'f'};
    auto row_layout = make_layout(make_shape(_2{}, _3{}), GenRowMajor());
    auto tensor = make_tensor(row_vec.data(), row_layout);

    print("Data is Row Major, Cute is Row major and Not transposed .\n");
    print("Cute Layout : ");
    print_layout(row_layout);
    print("\n");

    print("Data : ");
    print_char_tensor(tensor);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Data at 1-d coord %d and 2-d coord ", i);
        print(tensor.get_flat_coord(i));
        print("  : %c \n", tensor(i));
    }
    print("\n");
}

void test_data_row_cute_col_not_transposed()
{
    std::vector<char> row_vec{'a', 'b', 'c', 'd', 'e', 'f'};
    auto col_layout = make_layout(make_shape(_2{}, _3{}), GenColMajor());
    auto tensor = make_tensor(row_vec.data(), col_layout);

    print("Data is Row Major, Cute is Col major and Not transposed .\n");
    print("Cute Layout : ");
    print_layout(col_layout);
    print("\n");

    print("Data : ");
    print_char_tensor(tensor);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Data at 1-d coord %d and 2-d coord ", i);
        print(tensor.get_flat_coord(i));
        print("  : %c \n", tensor(i));
    }
    print("\n");
}

void test_data_col_cute_row_not_transposed()
{
    std::vector<char> col_vec{'a', 'd', 'b', 'e', 'c', 'f'};
    auto row_layout = make_layout(make_shape(_2{}, _3{}), GenRowMajor());
    auto tensor = make_tensor(col_vec.data(), row_layout);

    print("Data is Col Major , Cute is Row major and Not transposed .\n");
    print("Cute Layout : ");
    print_layout(row_layout);
    print("\n");

    print("Data : ");
    print_char_tensor(tensor);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Data at 1-d coord %d and 2-d coord ", i);
        print(tensor.get_flat_coord(i));
        print("  : %c \n", tensor(i));
    }
    print("\n");
}

void test_data_col_cute_col_not_transposed()
{
    std::vector<char> col_vec{'a', 'd', 'b', 'e', 'c', 'f'};
    auto col_layout = make_layout(make_shape(_2{}, _3{}), GenColMajor());
    auto tensor = make_tensor(col_vec.data(), col_layout);

    print("Data is Col Major , Cute is Col major and Not transposed .\n");
    print("Cute Layout : ");
    print_layout(col_layout);
    print("\n");

    print("Data : ");
    print_char_tensor(tensor);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Data at 1-d coord %d and 2-d coord ", i);
        print(tensor.get_flat_coord(i));
        print("  : %c \n", tensor(i));
    }
    print("\n");
}

void test_data_row_cute_row_transposed()
{
    std::vector<char> row_vec{'a', 'b', 'c', 'd', 'e', 'f'};
    auto row_layout = make_layout(make_shape(_3{}, _2{}), GenRowMajor());
    auto tensor = make_tensor(row_vec.data(), row_layout);

    print("Data is Row Major, Cute is Row major and transposed .\n");
    print("Cute Layout : ");
    print_layout(row_layout);
    print("\n");

    print("Data : ");
    print_char_tensor(tensor);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Data at 1-d coord %d and 2-d coord ", i);
        print(tensor.get_flat_coord(i));
        print("  : %c \n", tensor(i));
    }
    print("\n");
}

void test_data_row_cute_col_transposed()
{
    std::vector<char> row_vec{'a', 'b', 'c', 'd', 'e', 'f'};
    auto col_layout = make_layout(make_shape(_3{}, _2{}), GenColMajor());
    auto tensor = make_tensor(row_vec.data(), col_layout);

    print("Data is Row Major, Cute is Col major and transposed .\n");
    print("Cute Layout : ");
    print_layout(col_layout);
    print("\n");

    print("Data : ");
    print_char_tensor(tensor);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Data at 1-d coord %d and 2-d coord ", i);
        print(tensor.get_flat_coord(i));
        print("  : %c \n", tensor(i));
    }
    print("\n");
}

void test_data_col_cute_row_transposed()
{
    std::vector<char> col_vec{'a', 'd', 'b', 'e', 'c', 'f'};
    auto row_layout = make_layout(make_shape(_3{}, _2{}), GenRowMajor());
    auto tensor = make_tensor(col_vec.data(), row_layout);

    print("Data is Col Major , Cute is Row major and transposed .\n");
    print("Cute Layout : ");
    print_layout(row_layout);
    print("\n");

    print("Data : ");
    print_char_tensor(tensor);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Data at 1-d coord %d and 2-d coord ", i);
        print(tensor.get_flat_coord(i));
        print("  : %c \n", tensor(i));
    }
    print("\n");
}

void test_data_col_cute_col_transposed()
{
    std::vector<char> col_vec{'a', 'd', 'b', 'e', 'c', 'f'};
    auto col_layout = make_layout(make_shape(_3{}, _2{}), GenColMajor());
    auto tensor = make_tensor(col_vec.data(), col_layout);

    print("Data is Col Major , Cute is Col major and transposed .\n");
    print("Cute Layout : ");
    print_layout(col_layout);
    print("\n");

    print("Data : ");
    print_char_tensor(tensor);
    print("\n");

    for (int i{0}; i < size(tensor); ++i)
    {
        print("Data at 1-d coord %d and 2-d coord ", i);
        print(tensor.get_flat_coord(i));
        print("  : %c \n", tensor(i));
    }
    print("\n");
}

void test_data_row_cute_col_not_transposed_2d()
{
    std::vector<char> row_vec{'a', 'b', 'c', 'd', 'e', 'f'};
    auto col_layout = make_layout(make_shape(_2{}, _3{}), GenColMajor());
    auto tensor = make_tensor(row_vec.data(), col_layout);

    print("Data is Row Major, Cute is Col major and Not transposed .\n");
    print("Cute Layout : ");
    print_layout(col_layout);
    print("\n");

    print("Data : ");
    print_char_tensor(tensor);
    print("\n");

    for (int m{0}; m < size<0>(tensor); ++m)
        for (int n{0}; n < size<1>(tensor); ++n)
        {
            print("Data at 1-d coord (%d,%d)  : %c\n", m, n, tensor(m, n));
            // print(tensor.get_flat_coord(m,n));
            // print("  : %c \n", tensor(m,n));s
        }
    print("\n");
}

void test_single_tensor()
{
    // std::vector<int> vec{0};
    // auto tensor = make_tensor(vec.data(),make_shape(_1{}));

    int a = 10;
    auto tensor = make_tensor(&a,make_shape(_1{}));
    printf("%d",tensor(0));
}

int main()
{
    // test_access();
    // test_data_row_cute_row_not_transposed();
    // test_data_row_cute_col_not_transposed();
    // test_data_col_cute_row_not_transposed();
    // test_data_col_cute_col_not_transposed();

    // test_data_row_cute_row_transposed();
    // test_data_row_cute_col_transposed();
    // test_data_col_cute_row_transposed();
    // test_data_col_cute_col_transposed();
    // test_data_row_cute_col_not_transposed_2d();
    test_single_tensor();
}
