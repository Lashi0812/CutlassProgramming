#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"
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
    // test_swizzle<Swizzle<2, 0, 2>, Layout<Shape<_4, _4>, Stride<_4, _1>>>("Sw202_L4x4", ps);
    // test_swizzle<Swizzle<2, 0, 2>, Layout<Shape<_8, _4>, Stride<_4, _1>>>("Sw202_L8x4", ps);

    // test_swizzle<Swizzle<2, 0, 3>, Layout<Shape<_4, _8>, Stride<_8, _1>>>("Sw203_L4x8", ps);
    // test_swizzle<Swizzle<2, 0, 3>, Layout<Shape<_8, _8>, Stride<_8, _1>>>("Sw203_L8x8", ps);

    // test_swizzle<Swizzle<3, 0, 3>, Layout<Shape<_8, _8>, Stride<_8, _1>>>("Sw303_L8x8", ps);
    // test_swizzle<Swizzle<3, 0, 3>, Layout<Shape<_16, _8>, Stride<_8, _1>>>("Sw303_L16x8", ps);

    // test_swizzle<Swizzle<3, 0, 4>, Layout<Shape<_8, _16>, Stride<_16, _1>>>("Sw304_L8x16", ps);
    // test_swizzle<Swizzle<3, 0, 4>, Layout<Shape<_16, _16>, Stride<_16, _1>>>("Sw304_L16x16", ps);

    // test_swizzle<Swizzle<2, 1, 2>, Layout<Shape<_4, _4>, Stride<_4, _1>>>("Sw212_L4x4", ps);
    // test_swizzle<Swizzle<2, 1, 2>, Layout<Shape<_8, _4>, Stride<_4, _1>>>("Sw212_L8x4", ps);

    // test_swizzle<Swizzle<2, 1, 3>, Layout<Shape<_4, _8>, Stride<_8, _1>>>("Sw213_L4x8", ps);
    // test_swizzle<Swizzle<2, 1, 3>, Layout<Shape<_8, _8>, Stride<_8, _1>>>("Sw213_L8x8", ps);

    // test_swizzle<Swizzle<2, 2, 2>, Layout<Shape<_4, _16>, Stride<_16, _1>>>("Sw222_L4x16", ps);
    // test_swizzle<Swizzle<2, 2, 2>, Layout<Shape<_8, _16>, Stride<_16, _1>>>("Sw222_L8x16", ps);

    // test_swizzle<Swizzle<3, 3, 3>, Layout<Shape<_8, _64>, Stride<_64, _1>>>("Sw333_L8x64", ps);
    
    test_swizzle<Swizzle<2, 1, 5>, Layout<Shape<_16, _16>, Stride<_16, _1>>>("Sw215_L16x16", ps);
    test_swizzle<Swizzle<2, 2, 4>, Layout<Shape<_16, _16>, Stride<_16, _1>>>("Sw224_L16x16", ps);
    test_swizzle<Swizzle<2, 3, 3>, Layout<Shape<_16, _16>, Stride<_16, _1>>>("Sw233_L16x16", ps);
    test_swizzle<Swizzle<3, 1, 5>, Layout<Shape<_16, _16>, Stride<_16, _1>>>("Sw315_L16x16", ps);
    test_swizzle<Swizzle<3, 2, 4>, Layout<Shape<_16, _16>, Stride<_16, _1>>>("Sw324_L16x16", ps);
    test_swizzle<Swizzle<3, 3, 3>, Layout<Shape<_16, _16>, Stride<_16, _1>>>("Sw333_L16x16", ps);
    
    test_swizzle<Swizzle<2, 1, 5>, Layout<Shape<_4, _64>, Stride<_64, _1>>>("Sw215_L4x64_R", ps);
    test_swizzle<Swizzle<2, 2, 4>, Layout<Shape<_4, _64>, Stride<_64, _1>>>("Sw224_L4x64_R", ps);
    test_swizzle<Swizzle<2, 3, 3>, Layout<Shape<_4, _64>, Stride<_64, _1>>>("Sw233_L4x64_R", ps);
    test_swizzle<Swizzle<3, 1, 5>, Layout<Shape<_4, _64>, Stride<_64, _1>>>("Sw315_L4x64_R", ps);
    test_swizzle<Swizzle<3, 2, 4>, Layout<Shape<_4, _64>, Stride<_64, _1>>>("Sw324_L4x64_R", ps);
    test_swizzle<Swizzle<3, 3, 3>, Layout<Shape<_4, _64>, Stride<_64, _1>>>("Sw333_L4x64_R", ps);


    test_swizzle<Swizzle<2, 1, 5>, Layout<Shape<_4, _64>, Stride<_1,_4>>>("Sw215_L4x64_C", ps);
    test_swizzle<Swizzle<2, 2, 4>, Layout<Shape<_4, _64>, Stride<_1,_4>>>("Sw224_L4x64_C", ps);
    test_swizzle<Swizzle<2, 3, 3>, Layout<Shape<_4, _64>, Stride<_1,_4>>>("Sw233_L4x64_C", ps);
    test_swizzle<Swizzle<3, 1, 5>, Layout<Shape<_4, _64>, Stride<_1,_4>>>("Sw315_L4x64_C", ps);
    test_swizzle<Swizzle<3, 2, 4>, Layout<Shape<_4, _64>, Stride<_1,_4>>>("Sw324_L4x64_C", ps);
    test_swizzle<Swizzle<3, 3, 3>, Layout<Shape<_4, _64>, Stride<_1,_4>>>("Sw333_L4x64_C", ps);
    if (ps == 1)
        print_latex_footer();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
//                          Tile to Shape
////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_tile_to_shape() {
    auto comp = composition(Swizzle<2, 0, 2>{}, Layout<Shape<_4, _4>, Stride<_4, _1>>{});
    auto shape = tile_to_shape(comp, Shape<_8, _8>{});
    print_layout(shape);
}

int main(int argc, char *argv[]) {
    [[maybe_unused]] int ps{-1};
    if (argc >= 2)
        ps = atoi(argv[1]);
    test_swizzle_examples(ps);
    // test_tile_to_shape();
}