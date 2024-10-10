#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    int m = 256, n = 8192, k = 8192;
    std::srand(std::time(0));

    std::vector<int8_t> h_A(m * k);
    std::vector<int8_t> h_B(k * n);
    std::vector<int32_t> h_C(m * n);

    for (int i = 0; i < m * k; ++i)
    {
        h_A[i] = static_cast<int8_t>(std::rand() % 127);
    }
    for (int i = 0; i < k * n; ++i)
    {
        h_B[i] = static_cast<int8_t>(std::rand() % 127);
    }

    int8_t *A, *B;
    int32_t *C;
    cudaMalloc(&A, m * k * sizeof(int8_t));
    cudaMalloc(&B, k * n * sizeof(int8_t));
    cudaMalloc(&C, m * n * sizeof(int32_t));

    cudaMemcpy(A, h_A.data(), m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B.data(), k * n * sizeof(int8_t), cudaMemcpyHostToDevice);

    const int32_t alpha = 1;
    const int32_t beta = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_iters = 100;
    float total_time = 0.0f;

    for (int iter = 0; iter < num_iters; ++iter)
    {
        cudaEventRecord(start, 0);

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     m, n, k,
                     &alpha,
                     A, CUDA_R_8I, k,
                     B, CUDA_R_8I, k,
                     &beta,
                     C, CUDA_R_32I, m,
                     CUBLAS_COMPUTE_32I_PEDANTIC, CUBLAS_GEMM_DEFAULT);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }

    float avg_time = total_time / num_iters;
    float total_flops = 2.0 * m * n * k;
    std::cout << "Average cublasGemmEx execution time over " << num_iters << " runs: " << avg_time << " ms" << std::endl;
    std::cout << "Average TFLOPS: " << total_flops / (avg_time * 1e9) << std::endl;

    cudaMemcpy(h_C.data(), C, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
