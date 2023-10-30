#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <vector>
#include <iostream>

using namespace cute;
int main()
{
    auto layout = Layout<Shape<_2, _2>, Stride<_2, _1>>{};

    auto vec10 = std::vector{10, 11, 12, 13};
    auto ten10 = make_tensor(vec10.data(), layout);
    print_tensor(ten10);

    auto vec20 = std::vector{20, 21, 22, 23};
    auto ten20 = make_tensor(vec20.data(), layout);
    print_tensor(ten20);

    copy(ten10(_,1), ten20);
    print_tensor(ten10);
    print_tensor(ten20);

    copy(ten10(1,_), ten20);
    print_tensor(ten10);
    print_tensor(ten20);

    copy(ten10, ten20);
    print_tensor(ten10);
    print_tensor(ten20);  
}