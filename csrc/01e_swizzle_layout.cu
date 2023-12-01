#include "cute/layout.hpp"
#include "cute/swizzle.hpp"
#include <latex.hpp>

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
//                      Swizzle Layout
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Swizzle, typename Layout>
void test_swizzle(char const *test_name, int ps) {
    auto comp = composition(Swizzle{}, Layout{});
    custom_print(comp, test_name, ps);
}

void test_swizzle_examples(int ps) {
    if (ps == 1)
        print_latex_header();
    test_swizzle<Swizzle<2, 0, 2>, Layout<Shape<_4, _4>, Stride<_4, _1>>>("Sw202_L4x4", ps);
    test_swizzle<Swizzle<2, 0, 2>, Layout<Shape<_8, _4>, Stride<_4, _1>>>("Sw202_L8x4", ps);

    test_swizzle<Swizzle<2, 0, 3>, Layout<Shape<_4, _8>, Stride<_8, _1>>>("Sw203_L4x8", ps);
    test_swizzle<Swizzle<2, 0, 3>, Layout<Shape<_8, _8>, Stride<_8, _1>>>("Sw203_L8x8", ps);

    test_swizzle<Swizzle<3, 0, 3>, Layout<Shape<_8, _8>, Stride<_8, _1>>>("Sw303_L8x8", ps);
    test_swizzle<Swizzle<3, 0, 3>, Layout<Shape<_16, _8>, Stride<_8, _1>>>("Sw303_L16x8", ps);

    test_swizzle<Swizzle<3, 0, 4>, Layout<Shape<_8, _16>, Stride<_16, _1>>>("Sw304_L8x16", ps);
    test_swizzle<Swizzle<3, 0, 4>, Layout<Shape<_16, _16>, Stride<_16, _1>>>("Sw304_L16x16", ps);

    test_swizzle<Swizzle<2, 1, 2>, Layout<Shape<_4, _4>, Stride<_4, _1>>>("Sw212_L4x4", ps);
    test_swizzle<Swizzle<2, 1, 2>, Layout<Shape<_8, _4>, Stride<_4, _1>>>("Sw212_L8x4", ps);

    test_swizzle<Swizzle<2, 1, 3>, Layout<Shape<_4, _8>, Stride<_8, _1>>>("Sw213_L4x8", ps);
    test_swizzle<Swizzle<2, 1, 3>, Layout<Shape<_8, _8>, Stride<_8, _1>>>("Sw213_L8x8", ps);

    test_swizzle<Swizzle<2, 2, 2>, Layout<Shape<_4, _16>, Stride<_16, _1>>>("Sw222_L4x16", ps);
    test_swizzle<Swizzle<2, 2, 2>, Layout<Shape<_8, _16>, Stride<_16, _1>>>("Sw222_L8x16", ps);
    if (ps == 1)
        print_latex_footer();
}

int main(int argc, char *argv[]) {
    [[maybe_unused]] int ps{-1};
    if (argc >= 2)
        ps = atoi(argv[1]);
    test_swizzle_examples(ps);
}