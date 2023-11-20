#include <cute/layout.hpp>

using namespace cute;



void test_replace_front()
{
    auto layout = make_layout(make_shape(2, 2));
    print("Original shape");
    print(shape(layout));
    print("\n");
    print("Modified shape");
    print(replace_front(shape(layout), _));
    print("\n");
}

void test_replace_back()
{
    auto layout = make_layout(make_shape(2, 2));
    print("Original shape");
    print(shape(layout));
    print("\n");
    print("Modified shape");
    print(replace_back(shape(layout), _));
    print("\n");
}

void test_replace()
{
    auto layout = make_layout(make_shape(2, 2));
    print("Original shape");
    print(shape(layout));
    print("\n");
    print("Modified shape");
    print(replace<1>(shape(layout), _));
    print("\n");
}

int main()
{
    test_replace_front();
    test_replace_back();
    test_replace();
}