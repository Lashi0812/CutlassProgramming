#include <ATen/ATen.h>
#include <cute/tensor.hpp>
using namespace cute;

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

__global__ void fma(nv_half *A, nv_half *B, float *C)
{

    // IF you change the order of the register storage you will get the warp misaligned address issue
    float regD[4]{0., 0., 0., 0.};
    float regC[4];
    nv_half regA[4];
    nv_half regB[2];

    // copy A to reg
    auto tensorA = make_tensor(make_gmem_ptr(A),
                               make_layout(make_shape(_16{}, _8{})));
    auto each_threadA = zipped_divide(tensorA,
                                      make_tile(make_layout(_2{}, _1{}),
                                                make_layout(_2{}, _4{})))(_, threadIdx.x);
    auto dest_tensorA = make_tensor(make_rmem_ptr(regA), make_layout(_4{}, _1{}));

    // copy B to reg
    auto tensorB = make_tensor(make_gmem_ptr(B),
                               make_layout(make_shape(_8{}, _8{}), GenRowMajor()));
    auto each_threadB = zipped_divide(tensorB,
                                      make_tile(make_layout(_2{}, _1{})))(_, threadIdx.x);
    auto dest_tensorB = make_tensor(make_rmem_ptr(regB), make_layout(_2{}, _1{}));

    // uint32_t(&castRegA)[2] = reinterpret_cast<uint32_t(&)[2]>(regA);
    // uint32_t(&castRegB)[1] = reinterpret_cast<uint32_t(&)[1]>(regB);

    uint32_t *castRegA = reinterpret_cast<uint32_t *>(regA);
    uint32_t *castRegB = reinterpret_cast<uint32_t *>(regB);

    copy(each_threadA, dest_tensorA);
    copy(each_threadB, dest_tensorB);

    fma_asm(regC[0], regC[1], regC[2], regC[3],
            castRegA[0], castRegA[1],
            castRegB[0],
            regD[0], regD[1], regD[2], regD[3]);

    // copy reg to C
    auto tensorC = make_tensor(make_gmem_ptr(C),
                               make_layout(make_shape(_16{}, _8{})));
    auto each_threadC = zipped_divide(tensorC,
                                      make_tile(make_layout(_2{}, _1{}),
                                                make_layout(_2{}, _4{})))(_, threadIdx.x);
    auto dest_tensorC = make_tensor(make_rmem_ptr(regC), make_layout(_4{}, _1{}));

    copy(dest_tensorC, each_threadC);
}

int main()
{
    auto matA = at::rand({16, 8}, at::TensorOptions().dtype(at::kHalf));
    auto matB = at::rand({8, 8}, at::TensorOptions().dtype(at::kHalf));
    auto matC = at::rand({16, 8}, at::TensorOptions().dtype(at::kFloat));

    // auto matA = at::arange(16 * 8, at::TensorOptions().dtype(at::kHalf)).reshape({16, 8});
    // auto matB = at::arange(8 * 8, at::TensorOptions().dtype(at::kHalf)).reshape({8, 8});
    // auto matC = at::arange(16 * 8, at::TensorOptions().dtype(at::kFloat)).reshape({16, 8});

    nv_half *d_A, *d_B;
    float *d_C;

    cudaMalloc((void **)&d_A, matA.element_size() * matA.numel());
    cudaMalloc((void **)&d_B, matB.element_size() * matB.numel());
    cudaMalloc((void **)&d_C, matC.element_size() * matC.numel());

    cudaMemcpy(d_A, matA.data_ptr(), matA.element_size() * matA.numel(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matB.data_ptr(), matB.element_size() * matB.numel(), cudaMemcpyHostToDevice);

    fma<<<1, 32>>>(d_A, d_B, d_C);

    cudaMemcpy(matC.data_ptr(), d_C, matC.element_size() * matC.numel(), cudaMemcpyDeviceToHost);

    // std::cout << matC << std::endl;
    // std::cout << matA.matmul(matB) << std::endl;
    std::cout << (matC.to(at::kHalf).allclose((matA.matmul(matB))) ? "FMA Success" : "FMA Failed") << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
}
