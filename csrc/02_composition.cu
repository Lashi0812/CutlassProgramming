#include <cute/layout.hpp>
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

int main()
{
	// convert vector into matrix
	auto vector = make_shape(20);
	auto vector_stride = make_stride(2);
	auto vector_layout = make_layout(vector, vector_stride);
	print1D(vector_layout);

	auto matrix = make_shape(4, 5);

	auto col_major = make_stride(1, 4);
	auto matrix_col_layout = make_layout(matrix, col_major);
	auto composed_col_layout = composition(vector_layout, matrix_col_layout);
	print_layout(composed_col_layout);

	auto row_stride = make_stride(5, 1);
	auto matrix_row_layout = make_layout(matrix, row_stride);
	auto composed_row_layout = composition(vector_layout, matrix_row_layout);
	print_layout(composed_row_layout);

	// convert matrix into another matrix
	auto mat_a = make_layout(make_shape(20,2),make_stride(16,4));
	auto mat_b = make_layout(make_shape(4,5),make_stride(1,4));
	auto mat_c = composition(mat_a,mat_b);
	print_layout(mat_a);
	print_layout(mat_c);

}