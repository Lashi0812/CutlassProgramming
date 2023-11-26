#include <cute/algorithm/copy.hpp>

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
//                             Start of Copy traits
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename S, typename D>
void test_copy_trait()
{
    using copy_trait = Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<S, D>>;

    print("Thread layout : ");
    print(typename copy_trait::ThrID{});
    print("\n");

    print("Src layout : ");
    print_layout(typename copy_trait::SrcLayout{});
    print("\n");

    print("Dst layout : ");
    print_layout(typename copy_trait::DstLayout{});
    print("\n");

    print("Ref layout : ");
    print_layout(typename copy_trait::RefLayout{});
    print("\n");
}

void test_copy_trait_examples()
{
    {
        print("Int2 Bit Representation : \n");
        test_copy_trait<int2_t, int2_t>();
    }
    {
        print("Int8 Bit Representation : \n");
        test_copy_trait<int4_t, int4_t>();
    }
    {
        print("Int8 Bit Representation : \n");
        test_copy_trait<int8_t, int8_t>();
    }
    {
        print("Int16 Bit Representation : \n");
        test_copy_trait<int16_t, int16_t>();
    }
    {
        print("Int32 Bit Representation : \n");
        test_copy_trait<int, int>();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                              End of Copy traits
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
//                              Start of Copy Atom
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename S,typename D,typename T>
void test_copy_atom()
{
    using copy_atom = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<S,D>>,T>;

    print("Bit Source Layout : \n");
    print_latex(typename copy_atom::BitLayoutSrc{});
    print("\n");
    print("Value Source Layout : \n");
    print_latex(typename copy_atom::ValLayoutSrc{});
    print("\n");

    print("Bit Destination Layout : \n");
    print_latex(typename copy_atom::BitLayoutDst{});
    print("\n");
    print("Value Destination Layout : \n");
    print_latex(typename copy_atom::ValLayoutDst{});
    print("\n");

    print("Bit Reference Layout : \n");
    print_latex(typename copy_atom::BitLayoutRef{});
    print("\n");
    print("Value Reference Layout : \n");
    print_latex(typename copy_atom::ValLayoutRef{});
    print("\n");
}

void test_copy_atom_examples()
{
    {
        test_copy_atom<int,int,int8_t>();
    }
    {
        test_copy_atom<uint128_t,uint128_t,half_t>();
    }
    {
        test_copy_atom<uint128_t,uint128_t,int4_t>();
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//                              End of Copy Atom
////////////////////////////////////////////////////////////////////////////////////////////////////





int main()
{
    // test_copy_trait_examples();
    test_copy_atom_examples();
}