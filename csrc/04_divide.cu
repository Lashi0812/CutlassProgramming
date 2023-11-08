#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

template <class Shape, class Stride>
void print1D(Layout<Shape, Stride> const &layout)
{
	for (int i = 0; i < size(layout); ++i)
	{
		printf("%3d  ", layout(i));
	}
	printf("\n");
}

void test_logical_divide()
{
	auto vec = Layout<Int<16>, Int<3>>{};
	print1D(vec);
	auto col = Layout<_4, _2>{};
	print_layout(logical_divide(vec, col));
}

void test_zipped_divide_vs_logical_divide()
{
	auto tile = make_tile(make_layout(_2{}), make_layout(_3{}));
	auto tensor_layout = make_layout(make_shape(_8{}, Int<15>{}));

	auto tensor = make_counting_tensor(tensor_layout);
	print("Tensor : ");
	print_tensor(tensor);
	print("\n");

	

	auto log_divided = logical_divide(tensor, tile);
	print("Logical divided : ");
	print_tensor(log_divided);
	print("\n");


	auto zip_divided = zipped_divide(tensor, tile);
	print("Zipped Divided : ");
	print_tensor(zip_divided);
	print("\n");
}

void test_single_tensor()
{
	auto a = make_layout(make_shape(_2{},_2{}));
	print_layout(a);
	
	print("%d \n",a(0));
	print("%d \n",a(1));
	print("%d \n",a(2));
	print("%d \n",a(3));

	print("%d \n",a(0,0));
	print("%d \n",a(1,0));
	print("%d \n",a(0,1));
	print("%d \n",a(1,1));
}

int main()
{
	// test_logical_divide();
	// test_zipped_divide_vs_logical_divide();
	test_single_tensor();
}