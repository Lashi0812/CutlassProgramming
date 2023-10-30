#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <vector>
#include <iostream>

using namespace cute;
int main()
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
    std::cout << tensor(0,0) << std::endl;
    std::cout << tensor(0,1) << std::endl;
    std::cout << tensor(1,0) << std::endl;
    std::cout << tensor(1,1) << std::endl;
}
