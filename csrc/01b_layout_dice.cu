#include <cute/layout.hpp>

using namespace cute;

void test_dice()
{
    auto layout = make_layout(_4{}, _4{});
    print(layout);
    print("\n");
    print(dice(layout, Step<_0, _0>{}));
    print("\n");
    print(dice(layout, Step<_1, _1>{}));
    print("\n");
    print(dice(layout, Step<_0, _1>{}));
    print("\n");
    print(dice(layout, Step<_1, _0>{}));
    print("\n");
    print(dice(layout, Step<_1, Underscore>{}));
    print("\n");
}

void test_dice1()
{
    auto layout = make_layout(make_shape(_4{}, _4{}));
    print(layout);
    print("\n");
    print(dice(Step<_0, _0>{}, layout));
    print("\n");
    print(dice(Step<_1, _1>{}, layout));
    print("\n");
    print(dice(Step<_0, _1>{}, layout));
    print("\n");
    print(dice(Step<_1, _0>{}, layout));
    print("\n");
    print(dice(Step<_1, Underscore>{}, layout));
    print("\n");
}

int main()
{
    // test_dice();
    test_dice1();
}
