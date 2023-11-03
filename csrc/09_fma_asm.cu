#include <cute/numeric/int.hpp>

__device__ void
fma_asm(float &d0, float &d1, float &d2, float &d3,
        uint32_t const &a0, uint32_t const &a1,
        uint32_t const &b0,
        float const &c0, float const &c1, float const &c2, float const &c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1),
          "r"(b0),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ void fma()
{
    float d0{0}, d1{0}, d2{0}, d3{0};
    float a32[4] = {1., 1., 1., 1.};
    float b32[2] = {1., 1.};
    uint32_t A[2];
    uint32_t B[1];

    asm("cvt.rz.f16x2.f32 %0, %1, %2;\n" : "=r"(A[0]) : "f"(a32[0]), "f"(a32[1]));
    asm("cvt.rz.f16x2.f32 %0, %1, %2;\n" : "=r"(A[1]) : "f"(a32[2]), "f"(a32[3]));
    // B
    asm("cvt.rz.f16x2.f32 %0, %1, %2;\n" : "=r"(B[0]) : "f"(b32[0]), "f"(b32[1]));
    //        1                 1
    // 0011110000000000 0011110000000000
    // int(00111100000000000011110000000000,base=2) = 1006648320
    cute::uint32_t a0{1006648320}; // 1,1
    cute::uint32_t a1{1006648320}; // 1,1
    cute::uint32_t b0{1006648320}; // 1,1
    float c0{0}, c1{0}, c2{0}, c3{0};

    fma_asm(d0, d1, d2, d3,
            A[0], A[1],
            B[0],
            // a0, a1,
            // b0,
            c0, c1, c2, c3);

    printf("%f ,%f, %f,%f\n", d0, d1, d2, d3);
}

int main()
{
    fma<<<1, 32>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
}