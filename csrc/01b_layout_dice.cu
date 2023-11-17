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

int main()
{
    test_dice();
}
