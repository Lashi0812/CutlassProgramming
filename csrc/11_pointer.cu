#include <cute/tensor.hpp>
#include <vector>

using namespace cute;

void test_gmem_ptr()
{
    auto vec1 = std::vector<int>{1, 2};
    auto vec2 = std::vector<int>{1, 2};

    auto ptr1 = make_gmem_ptr(vec1.data());
    auto ptr2 = make_gmem_ptr(vec2.data());
    auto ptr3 = make_smem_ptr(vec1.data());

    print(ptr1);
    print("\n");
    print(ptr2);
    print("\n");
    std::cout << "Is ptr1 is gmem ptr : " << is_gmem<decltype(ptr1)>::value << std::endl;
    std::cout << "Is ptr2 is gmem ptr : " << is_gmem<decltype(ptr2)>::value << std::endl;
    std::cout << "Is ptr3 is gmem ptr : " << is_gmem<decltype(ptr3)>::value << std::endl;
    print("\nComparing address : \n");
    std::cout << "\tptr1 == ptr2 : " << ((ptr1 == ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\tptr1 != ptr2 : " << ((ptr1 != ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\tptr1  < ptr2 : " << ((ptr1 < ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\tptr1  > ptr2 : " << ((ptr1 > ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\tptr1 >= ptr2 : " << ((ptr1 >= ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\tptr1 <= ptr2 : " << ((ptr1 <= ptr2) ? "Yes" : "No") << std::endl;
    print("\nComparing value in the address :\n");
    std::cout << "\t*ptr1 == *ptr2 : " << ((*ptr1 == *ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\t*ptr1 != *ptr2 : " << ((*ptr1 != *ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\t*ptr1  < *ptr2 : " << ((*ptr1 < *ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\t*ptr1  > *ptr2 : " << ((*ptr1 > *ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\t*ptr1 >= *ptr2 : " << ((*ptr1 >= *ptr2) ? "Yes" : "No") << std::endl;
    std::cout << "\t*ptr1 <= *ptr2 : " << ((*ptr1 <= *ptr2) ? "Yes" : "No") << std::endl;
    print("\nMoving the pointer address : ");
    print(ptr1 + 1);
    print("\n");
    print("\nMoving the pointer address and dereferencing: ");
    print(ptr1[1]);
    print("\n");
}

int main()
{
    test_gmem_ptr();
}