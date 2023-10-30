#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
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
    auto vec = Layout<Int<16>,Int<3>>{};
    print1D(vec);
    auto col = Layout<_4,_2>{};
    print_layout(logical_divide(vec,col));
}