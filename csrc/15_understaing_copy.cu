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




int main()
{
    test_copy_trait_examples();
}