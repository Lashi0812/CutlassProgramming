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
	auto a = make_layout(make_shape(_2{}, _2{}));
	print_layout(a);

	print("%d \n", a(0));
	print("%d \n", a(1));
	print("%d \n", a(2));
	print("%d \n", a(3));

	print("%d \n", a(0, 0));
	print("%d \n", a(1, 0));
	print("%d \n", a(0, 1));
	print("%d \n", a(1, 1));
}

void test_tile_vs_shape()
{
	auto tile = make_tile(make_layout(_2{}), make_layout(_3{}));
	auto shape = make_shape(_2{}, _3{});
	auto tensor_layout = make_layout(make_shape(_8{}, Int<15>{}));

	auto tensor = make_counting_tensor(tensor_layout);
	print("Tensor : ");
	print_tensor(tensor);
	print("\n");

	auto zip_tile_divided = zipped_divide(tensor, tile);
	print("Zipped Tiled Divided : ");
	print_tensor(zip_tile_divided);
	print("\n");

	auto zip_shape_divided = zipped_divide(tensor, shape);
	print("Zipped shape Divided : ");
	print_tensor(zip_shape_divided);
	print("\n");
}

void test_zipped_div_1d()
{
	auto layout = make_layout(make_shape(4, 2), make_stride(8, 1));
	auto zip_div = zipped_divide(layout, make_shape(2));
	print("Zipped shape Divided : ");
	print_layout(zip_div);
	print("\n");
	print_layout(zip_div(make_coord(make_coord(0), make_coord(_, _))));
}

template <typename TensorLayout, typename ThreadLayout, typename ThreadStep>
void test_local_part_1d(TensorLayout layout, ThreadLayout thread_layout, ThreadStep)
{
	// auto layout = make_layout(make_shape(4, 2));
	// auto thread_layout = make_layout(make_shape(2, 4));
	auto tensor = make_counting_tensor(layout);
	print_tensor(tensor);
	for (int i{0}; i < size(thread_layout); ++i)
	{
		auto part = local_partition(tensor, thread_layout, i, ThreadStep{});
		print("Partition for Thread %d : ", i);
		print_tensor(part);
		print("\n");
	}
}

int main()
{
	// test_logical_divide();
	// test_zipped_divide_vs_logical_divide();
	// test_single_tensor();
	// test_tile_vs_shape();
	// test_zipped_div_1d();
	{
		auto layout = make_layout(make_shape(4, 2));
		auto thread_layout = make_layout(make_shape(2, 4));
		test_local_part_1d(layout, thread_layout, Step<_1, Underscore>{});
	}
	{
		auto layout = make_layout(make_shape(4, 2));
		auto thread_layout = make_layout(make_shape(4, 2));
		test_local_part_1d(layout, thread_layout, Step<Underscore, _1>{});
	}
}