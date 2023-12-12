#pragma once

#include <cute/config.hpp>

#include <cute/underscore.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>

namespace cute {

template <typename Args>
void custom_print(Args args, const char *name, int ps = -1, int hf = 0) {
    if (ps == 0)
        print_layout(args);
    else if (ps == 1)
        print_latex(args, name, hf);
    else
        print(args);
}

//////////////////////////////////////////////////////////////////////////////////////////
//                              Header and footer
//////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE void print_latex_header() {
    printf("\\documentclass{standalone}\n"
           "\\usepackage{tikz}\n"
           "\\usetikzlibrary{external}\n"
           "\\tikzexternalize[mode=list and make]\n\n"
           "\\newcommand{\\setfilename}[1]{\n"
           "\\tikzsetnextfilename{#1}}\n"

           "\\tikzset{\n"
           "png export/.style={\n"
           "external/system call/.add={}{; convert -density 300 -background \"grey\" -alpha remove "
           "\"\\image.pdf\" "
           "\"$(IMAGE_PATH)/\\image.png\"},\n"
           "/pgf/images/external info,\n"
           "/pgf/images/include external/.code={\n"
           "\\includegraphics[width=\\pgfexternalwidth,height=\\pgfexternalheight]{##1.png}\n"
           "},\n}\n}\n\n"

           "\\begin{document}\n");
}

CUTE_HOST_DEVICE void print_latex_footer() { printf("\\end{document}\n"); }

//////////////////////////////////////////////////////////////////////////////////////////
//                              Layouts
//////////////////////////////////////////////////////////////////////////////////////////

// hf == 0 only picture
// hf == 1 header +  picture
// hf == 2 picture + footer
// hf == 3 header +  picture + footer

// Generic 2D Layout to Latex printer -- B&W 8-value color coding
template <class Layout>
CUTE_HOST_DEVICE void
print_latex(Layout const &layout, const char *name, int hf = 0) // (m,n) -> idx
{
    CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

    // char const *latex_header =
    //   "\\tikzset{png export}"
    //   "\\setfilename{test-Custom1}"
    //   "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
    //   ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center,font=\\Large}]\n\n";
    // char const *latex_footer = "\\end{tikzpicture}\n";

    char const *color_map[8] = {
      "black!00",
      "black!40",
      "black!20",
      "black!60",
      "black!10",
      "black!50",
      "black!30",
      "black!70"};

    // Header
    printf("%% Layout: ");
    print(layout);
    printf("\n");

    if (hf & 1)
        print_latex_header();

    printf(
      "\\tikzset{png export}\n"
      "\\setfilename{%s}\n"
      "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
      ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n",
      name);

    // Layout
    for (int i = 0; i < size<0>(layout); ++i) {
        for (int j = 0; j < size<1>(layout); ++j) {
            int idx = layout(i, j);

            printf("\\node[box,fill=%s] at (%d,%d) {%d};\n", color_map[idx % 8], i, j, idx);
        }
    }

    // Labels
    for (int i = 0, j = -1; i < size<0>(layout); ++i) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
    }
    for (int j = 0, i = -1; j < size<1>(layout); ++j) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
    }

    // Footer
    printf("\\node[above, font=\\bfseries, anchor=west] at ([yshift=8mm] current bounding "
           "box.north west) "
           "{\\textbf{\\detokenize{");
    print(name);
    print("\n");
    print(layout);
    printf("}}};\n");
    printf("\\end{tikzpicture}\n");

    if ((hf >> 1) & 1)
        print_latex_footer();
}

// Generic ThrVal 2D Layout to Latex TIKZ -- 8-value color coded by thread
template <class Layout, class ThrID>
CUTE_HOST_DEVICE void print_latex(
  Layout const &layout,
  ThrID const &thr,
  const char *name,
  int hf = 0) // (m,n) -> (tid,vid)  and  tid -> thr_idx
{
    CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

    // char const *latex_header =
    //   "\\documentclass[convert]{standalone}\n"
    //   "\\usepackage{tikz}\n\n"
    //   "\\begin{document}\n"
    //   "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
    //   ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n";
    char const *latex_footer = "\\end{tikzpicture}\n";

    char const *color_map[8] = {
      "{rgb,255:red,175;green,175;blue,255}",
      "{rgb,255:red,175;green,255;blue,175}",
      "{rgb,255:red,255;green,255;blue,175}",
      "{rgb,255:red,255;green,175;blue,175}",
      "{rgb,255:red,210;green,210;blue,255}",
      "{rgb,255:red,210;green,255;blue,210}",
      "{rgb,255:red,255;green,255;blue,210}",
      "{rgb,255:red,255;green,210;blue,210}"};

    // Header
    printf("%% layout: ");
    print(layout);
    printf("\n");
    printf("%% thrid:  ");
    print(thr);
    printf("\n\n");

    if (hf & 1)
        print_latex_header();

    printf(
      "\\tikzset{png export}\n"
      "\\setfilename{%s}\n"
      "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
      ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n",
      name);

    // Layout
    for (int i = 0; i < size<0>(layout); ++i) {
        for (int j = 0; j < size<1>(layout); ++j) {
            int thrid = layout(i, j) % size(thr);
            int val_idx = layout(i, j) / size(thr);
            int thr_idx = thr(thrid);

            printf(
              "\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color_map[thr_idx % 8],
              i,
              j,
              thr_idx,
              val_idx);
        }
    }

    // Labels
    for (int i = 0, j = -1; i < size<0>(layout); ++i) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
    }
    for (int j = 0, i = -1; j < size<1>(layout); ++j) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
    }

    // Footer
    printf(latex_footer);
    if ((hf >> 1) & 1)
        print_latex_footer();
}

//////////////////////////////////////////////////////////////////////////////////////////
//                              copy_atom
//////////////////////////////////////////////////////////////////////////////////////////

template <class... Args>
CUTE_HOST_DEVICE auto print_latex(TiledCopy<Args...> const &copy, const char *name, int hf = 0) {
    auto [layoutS_MN, thrID_S] = copy.get_layoutS_MN();
    auto [layoutD_MN, thrID_D] = copy.get_layoutD_MN();

    print_latex_copy(layoutS_MN, thrID_S, layoutD_MN, thrID_D, name, hf);
}

// MNK Copy Layout to Latex TIKZ -- 8-value color coded by thread
template <class LayoutS, class ThrIDS, class LayoutD, class ThrIDD>
CUTE_HOST_DEVICE void print_latex_copy(
  LayoutS const &S,
  ThrIDS const &TS, // (m,n) -> (tid,vid)  and  tid -> thr_idx
  LayoutD const &D,
  ThrIDD const &TD,
  const char *name,
  int hf = 0) // (m,n) -> (tid,vid)  and  tid -> thr_idx
{
    CUTE_STATIC_ASSERT_V(rank(S) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<2>{});

    assert(size<0>(S) == size<0>(D));
    assert(size<1>(S) == size<1>(D));

    //   char const* latex_header =
    //       "\\documentclass{standalone}\n"
    //       "\\usepackage{tikz}\n"
    //       "\\usetikzlibrary{external}\n"
    //       "\\tikzexternalize\n"
    //       "\\begin{document}\n"
    //       "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/.style={rectangle,draw=black,thick,minimum
    //       size=1cm,anchor=center}]\n\n";
    //   char const* latex_footer =
    //       "\\end{tikzpicture}\n"
    //       "\\end{document}\n";

    char const *color_map[8] = {
      "{rgb,255:red,175;green,175;blue,255}",
      "{rgb,255:red,175;green,255;blue,175}",
      "{rgb,255:red,255;green,255;blue,175}",
      "{rgb,255:red,255;green,175;blue,175}",
      "{rgb,255:red,210;green,210;blue,255}",
      "{rgb,255:red,210;green,255;blue,210}",
      "{rgb,255:red,255;green,255;blue,210}",
      "{rgb,255:red,255;green,210;blue,210}",
    };

    // Header
    printf("%% LayoutS: ");
    print(S);
    printf("\n");
    printf("%% ThrIDS : ");
    print(TS);
    printf("\n");
    printf("%% LayoutD: ");
    print(D);
    printf("\n");
    printf("%% ThrIDD : ");
    print(TD);
    printf("\n\n");

    if (hf & 1)
        print_latex_header();

    printf(
      "\\tikzset{png export}\n"
      "\\setfilename{%s}\n"
      "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
      ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n",
      name);

    // S starting at 0,0
    for (int i = 0; i < size<0>(S); ++i) {
        for (int j = 0; j < size<1>(S); ++j) {
            int thrid = S(i, j) % size(TS);
            int val_idx = S(i, j) / size(TS);
            int thr_idx = TS(thrid);

            printf(
              "\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color_map[thr_idx % 8],
              i,
              j,
              thr_idx,
              val_idx);
        }
    }

    // D starting at 0,size<1>(S)+3
    for (int i = 0; i < size<0>(D); ++i) {
        for (int j = 0; j < size<1>(D); ++j) {
            int thrid = D(i, j) % size(TD);
            int val_idx = D(i, j) / size(TD);
            int thr_idx = TD(thrid);

            printf(
              "\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color_map[thr_idx % 8],
              i,
              j + size<1>(S) + 3,
              thr_idx,
              val_idx);
        }
    }

    // S Labels
    for (int i = 0, j = -1; i < size<0>(S); ++i) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
    }
    for (int j = 0, i = -1; j < size<1>(S); ++j) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
    }
    // D Labels
    for (int i = 0, j = size<1>(D); i < size<0>(S); ++i) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j + size<1>(S) + 3, i);
    }
    for (int j = 0, i = -1; j < size<1>(D); ++j) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j + size<1>(S) + 3, j);
    }

    // Footer
    printf("\\node[above, font=\\bfseries,  align=left] at ([yshift=4mm] current bounding "
           "box.north west) "
           "{");
    // clang-format off
    print("\\textbf{\\detokenize{");print(name);print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutS   ");print(S   );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutTS  ");print(TS  );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutD   ");print(D   );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutTD  ");print(TD  );print("}}\\\\\n");
    // clang-format on
    printf("};\n");
    printf("\\end{tikzpicture}\n");
    if ((hf >> 1) & 1)
        print_latex_footer();
}

//////////////////////////////////////////////////////////////////////////////////////////
//                              mma_atom
//////////////////////////////////////////////////////////////////////////////////////////

template <class... Args>
CUTE_HOST_DEVICE auto print_latex(TiledMMA<Args...> const &mma, const char *name, int hf = 0) {
    auto layout_and_thrid_C = mma.get_layoutC_MN();
    auto layoutC_MN = get<0>(layout_and_thrid_C);
    auto thrID_C = get<1>(layout_and_thrid_C);

    auto layout_and_thrid_A = mma.get_layoutA_MK();
    auto layoutA_MK = get<0>(layout_and_thrid_A);
    auto thrID_A = get<1>(layout_and_thrid_A);

    auto layout_and_thrid_B = mma.get_layoutB_NK();
    auto layoutB_NK = get<0>(layout_and_thrid_B);
    auto thrID_B = get<1>(layout_and_thrid_B);

    print_latex_mma(layoutC_MN, thrID_C, layoutA_MK, thrID_A, layoutB_NK, thrID_B, name, hf);
}

// EXPERIMENTAL -- Doesn't work with Swizzled Thr TileMMAs...
template <class... Args>
CUTE_HOST_DEVICE auto print_latex_2(TiledMMA<Args...> const &mma, const char *name, int hf = 0) {
    print_latex_mma(
      typename TiledMMA<Args...>::TiledShape_MNK{},
      mma.get_layoutC_TV(),
      mma.get_layoutA_TV(),
      mma.get_layoutB_TV(),
      name,
      hf);
}

// MNK MMA Layout to Latex TIKZ -- 8-value color coded by thread
template <class LayoutC, class ThrIDC, class LayoutA, class ThrIDA, class LayoutB, class ThrIDB>
CUTE_HOST_DEVICE void print_latex_mma(
  LayoutC const &C,
  ThrIDC const &TC, // (m,n) -> (tid,vid)  and  tid -> thr_idx
  LayoutA const &A,
  ThrIDA const &TA, // (m,k) -> (tid,vid)  and  tid -> thr_idx
  LayoutB const &B,
  ThrIDB const &TB, // (n,k) -> (tid,vid)  and  tid -> thr_idx
  const char *name,
  int hf = 0) {
    CUTE_STATIC_ASSERT_V(rank(C) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(A) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(B) == Int<2>{});

    assert(size<0>(A) == size<0>(C));
    assert(size<0>(B) == size<1>(C));
    assert(size<1>(A) == size<1>(B));

    // char const *latex_header =
    //   "\\documentclass{standalone}\n"
    //   "\\usepackage{tikz}\n"
    //   "\\usetikzlibrary{external}\n"
    //   "\\tikzexternalize\n"
    //   "\\begin{document}\n"
    //   "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
    //   ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n";
    // char const *latex_footer = "\\end{tikzpicture}\n"
    //                            "\\end{document}\n";

    char const *color_map[8] = {
      "{rgb,255:red,175;green,175;blue,255}",
      "{rgb,255:red,175;green,255;blue,175}",
      "{rgb,255:red,255;green,255;blue,175}",
      "{rgb,255:red,255;green,175;blue,175}",
      "{rgb,255:red,210;green,210;blue,255}",
      "{rgb,255:red,210;green,255;blue,210}",
      "{rgb,255:red,255;green,255;blue,210}",
      "{rgb,255:red,255;green,210;blue,210}"};

    // Header
    printf("%% LayoutC: ");
    print(C);
    printf("\n");
    printf("%% ThrIDC : ");
    print(TC);
    printf("\n");
    printf("%% LayoutA: ");
    print(A);
    printf("\n");
    printf("%% ThrIDA : ");
    print(TA);
    printf("\n");
    printf("%% LayoutB: ");
    print(B);
    printf("\n");
    printf("%% ThrIDB : ");
    print(TB);
    printf("\n\n");

    if (hf & 1)
        print_latex_header();

    printf(
      "\\tikzset{png export}\n"
      "\\setfilename{%s}\n"
      "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
      ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n",
      name);

    // C starting at 0,0
    for (int m = 0; m < size<0>(C); ++m) {
        for (int n = 0; n < size<1>(C); ++n) {
            int thrid = C(m, n) % size(TC);
            int val_idx = C(m, n) / size(TC);
            int thr_idx = TC(thrid);

            printf(
              "\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color_map[thr_idx % 8],
              m,
              n,
              thr_idx,
              val_idx);
        }
    }

    // A starting at 0,-size<1>(A)-1
    for (int m = 0; m < size<0>(A); ++m) {
        for (int k = 0; k < size<1>(A); ++k) {
            int thrid = A(m, k) % size(TA);
            int val_idx = A(m, k) / size(TA);
            int thr_idx = TA(thrid);

            printf(
              "\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color_map[thr_idx % 8],
              m,
              k - 1 - size<1>(A),
              thr_idx,
              val_idx);
        }
    }

    // B starting at -size<1>(B)-1,0
    for (int n = 0; n < size<0>(B); ++n) {
        for (int k = 0; k < size<1>(B); ++k) {
            int thrid = B(n, k) % size(TB);
            int val_idx = B(n, k) / size(TB);
            int thr_idx = TB(thrid);

            printf(
              "\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color_map[thr_idx % 8],
              k - 1 - size<1>(B),
              n,
              thr_idx,
              val_idx);
        }
    }

    // A labels
    for (int m = 0, k = -1; m < size<0>(A); ++m) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k - 1 - size<1>(A), m);
    }
    for (int k = 0, m = -1; k < size<1>(A); ++k) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k - 1 - size<1>(A), k);
    }
    // B labels
    for (int n = 0, k = -1; n < size<0>(B); ++n) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k - 1 - size<1>(B), n, n);
    }
    for (int k = 0, n = -1; k < size<1>(B); ++k) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k - 1 - size<1>(B), n, k);
    }

    // Footer
    printf("\\node[above, font=\\bfseries, align=left] at ([yshift=4mm] current "
           "bounding "
           "box.north west) "
           "{");
    // clang-format off

    print("\\textbf{\\detokenize{"           );print(name );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutC"    );print(C    );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutThrC" );print(TC   );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutA"    );print(A    );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutThrA" );print(TA   );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutB"    );print(B    );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutThrB" );print(TB   );print("}}\\\\\n");
    // clang-format on
    printf("};\n");
    printf("\\end{tikzpicture}\n");
    if ((hf >> 1) & 1)
        print_latex_footer();
}

// ThrVal MMA Layout to Latex TIKZ -- 8-value color coded by thread
template <class Shape_MNK, class LayoutC, class LayoutA, class LayoutB>
CUTE_HOST_DEVICE void print_latex_mma(
  Shape_MNK const &shape_mnk,
  LayoutC const &C, // (thr_idx,vid) -> (m,n)
  LayoutA const &A, // (thr_idx,vid) -> (m,k)
  LayoutB const &B, // (thr_idx,vid) -> (n,k)
  const char *name,
  int hf = 0) {
    CUTE_STATIC_ASSERT_V(rank(C) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(A) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(B) == Int<2>{});

    // char const *latex_header =
    //   "\\documentclass{standalone}\n"
    //   "\\usepackage{tikz}\n"
    //   "\\usetikzlibrary{external}\n"
    //   "\\tikzexternalize\n"
    //   "\\begin{document}\n"
    //   "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
    //   ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n";
    // char const *latex_footer = "\\end{tikzpicture}\n"
    //                            "\\end{document}\n";

    char const *color_map[8] = {
      "{rgb,255:red,175;green,175;blue,255}",
      "{rgb,255:red,175;green,255;blue,175}",
      "{rgb,255:red,255;green,255;blue,175}",
      "{rgb,255:red,255;green,175;blue,175}",
      "{rgb,255:red,210;green,210;blue,255}",
      "{rgb,255:red,210;green,255;blue,210}",
      "{rgb,255:red,255;green,255;blue,210}",
      "{rgb,255:red,255;green,210;blue,210}"};

    // Header
    printf("%% Shape_MNK: ");
    print(shape_mnk);
    printf("\n");
    printf("%% LayoutC  : ");
    print(C);
    printf("\n");
    printf("%% LayoutA  : ");
    print(A);
    printf("\n");
    printf("%% LayoutB  : ");
    print(B);
    printf("\n\n");

    if (hf & 1)
        print_latex_header();

    printf(
      "\\tikzset{png export}\n"
      "\\setfilename{%s}\n"
      "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
      ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n",
      name);

    auto M = size<0>(shape_mnk);
    auto N = size<1>(shape_mnk);
    auto K = size<2>(shape_mnk);

    // C starting at 0,0
    bool c_filled[M][N] = {};
    for (int t = 0; t < size<0>(C); ++t) {
        for (int v = 0; v < size<1>(C); ++v) {
            int m = C(t, v) % M;
            int n = C(t, v) / M;

            if (not c_filled[m][n]) {
                printf(
                  "\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
                  color_map[t % 8],
                  m,
                  n,
                  t,
                  v);
                c_filled[m][n] = true;
            }
        }
    }

    // A starting at 0,-size<1>(A)-1
    bool a_filled[M][K] = {};
    for (int t = 0; t < size<0>(A); ++t) {
        for (int v = 0; v < size<1>(A); ++v) {
            int m = A(t, v) % M;
            int k = A(t, v) / M;

            if (not a_filled[m][k]) {
                printf(
                  "\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
                  color_map[t % 8],
                  m,
                  k - 1 - K,
                  t,
                  v);
                a_filled[m][k] = true;
            }
        }
    }

    // B starting at -size<1>(B)-1,0
    bool b_filled[N][K] = {};
    for (int t = 0; t < size<0>(B); ++t) {
        for (int v = 0; v < size<1>(B); ++v) {
            int n = B(t, v) % N;
            int k = B(t, v) / N;

            if (not b_filled[n][k]) {
                printf(
                  "\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
                  color_map[t % 8],
                  k - 1 - K,
                  n,
                  t,
                  v);
                b_filled[n][k] = true;
            }
        }
    }

    // A labels
    for (int m = 0, k = -1; m < M; ++m) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k - 1 - K, m);
    }
    for (int k = 0, m = -1; k < K; ++k) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k - 1 - K, k);
    }
    // B labels
    for (int n = 0, k = -1; n < N; ++n) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k - 1 - K, n, n);
    }
    for (int k = 0, n = -1; k < K; ++k) {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k - 1 - K, n, k);
    }

    // Footer
    printf("\\node[above, font=\\bfseries, align=left] (title) at (current "
           "bounding box.north west) {");
    // clang-format off
    print("\\textbf{\\detokenize{"        );print(name );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutC" );print(C    );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutA" );print(A    );print("}}\\\\\n");
    print("\\textbf{\\detokenize{LayoutB" );print(B    );print("}}\\\\\n");
    // clang-format on
    printf("};\n");
    printf("\\end{tikzpicture}\n");

    if ((hf >> 1) & 1)
        print_latex_footer();
}

} // namespace cute