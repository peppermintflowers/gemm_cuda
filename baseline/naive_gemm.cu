#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// Checks for and prints error encountered
#define CUDA_CHECK(call)                           \
    do                                             \
    {                                              \
        cudaError_t error = (call);                \
        if (error != cudaSuccess)                  \
        {                                          \
            std::cerr << "CUDA Error: "            \
                      << cudaGetErrorString(error) \
                      << " at " << __FILE__        \
                      << ":" << __LINE__           \
                      << std::endl;                \
            std::exit(EXIT_FAILURE);               \
        }                                          \
    } while (0)

constexpr int WARMUP_RUNS = 3;
constexpr int MEASUREMENTS = 10;

/*
 * Naive computation for C = A x B
 */
__global__ void naive_gemm(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    // Global column index for thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Global row index for thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N)
    {
        int A_i = N * row;
        int C_ij = N * row + col;

        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            int A_ik = A_i + k;       // i,k
            int B_kj = N * k + col;   // k,j

            sum += A[A_ik] * B[B_kj];
        }

        C[C_ij] = sum;
    }
}

/*
 * Calculate median runtime
 */
float median(std::vector<float> runtimes)
{
    std::sort(
        runtimes.begin(),
        runtimes.end()
    );

    return (
        runtimes[MEASUREMENTS / 2 - 1] +
        runtimes[MEASUREMENTS / 2]
    ) / 2.0f;
}

int main()
{
    const int N = 8192;

    const size_t num_elements =
        static_cast<size_t>(N) * N;

    const size_t bytes =
        num_elements * sizeof(float);

    std::vector<float> A(
        num_elements,
        1.0f
    );

    std::vector<float> B(
        num_elements,
        2.0f
    );

    std::vector<float> C(
        num_elements,
        0.0f
    );

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;

    CUDA_CHECK(cudaMalloc(
        &dA,
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        &dB,
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        &dC,
        bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        dA,
        A.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB,
        B.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    dim3 block(32, 8);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    // Warmup runs
    for (int run = 0; run < WARMUP_RUNS; run++)
    {
        naive_gemm<<<grid, block>>>(
            dA,
            dB,
            dC,
            N
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> runtimes;
    runtimes.reserve(MEASUREMENTS);

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Measure naive GEMM runtime
    for (int run = 0; run < MEASUREMENTS; run++)
    {
        CUDA_CHECK(cudaEventRecord(start));

        naive_gemm<<<grid, block>>>(
            dA,
            dB,
            dC,
            N
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        ));

        runtimes.push_back(milliseconds);
    }

    const float median_ms =
        median(runtimes);

    // GEMM performs approximately 2*N^3 floating-point operations
    const double flops =
        2.0 *
        static_cast<double>(N) *
        N *
        N;

    // Convert milliseconds to seconds
    const double seconds =
        median_ms / 1000.0;

    const double gflops =
        flops / seconds / 1e9;

    std::cout
        << "N: "
        << N
        << "\n";

    std::cout
        << "Median Time: "
        << median_ms
        << " ms\n";

    std::cout
        << "Performance: "
        << gflops
        << " GFLOP/s\n";

    // Copy result back to cpu to verify correctness
    CUDA_CHECK(cudaMemcpy(
        C.data(),
        dC,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // A contains 1.0 and B contains 2.0, so every output is 2*N
    const float expected =
        2.0f * N;

    bool passed = true;

    for (size_t i = 0; i < num_elements; i++)
    {
        if (C[i] != expected)
        {
            passed = false;

            std::cout
                << "Mismatch at index "
                << i
                << ": expected "
                << expected
                << ", got "
                << C[i]
                << "\n";

            break;
        }
    }

    std::cout
        << "Correctness: "
        << (passed ? "PASS" : "FAIL")
        << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}