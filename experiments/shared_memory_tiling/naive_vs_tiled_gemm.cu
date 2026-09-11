#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

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
constexpr int BLOCK_SIZE = 16;

/*
 * Naive GEMM.
 * One thread computes one element of C directly from global memory.
 */
__global__ void naive_gemm(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int col =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int row =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            sum +=
                A[row * N + k] *
                B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

/*
 * Tiled GEMM.
 * Uses the same 16x16 thread-block shape as the naive kernel,
 * but stages A and B tiles through shared memory.
 */
__global__ void tiled_gemm(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int col =
        blockIdx.x * BLOCK_SIZE +
        threadIdx.x;

    int row =
        blockIdx.y * BLOCK_SIZE +
        threadIdx.y;

    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    for (
        int tile = 0;
        tile < N;
        tile += BLOCK_SIZE)
    {
        int A_col =
            tile + threadIdx.x;

        int B_row =
            tile + threadIdx.y;

        // Cooperatively load A tile
        if (row < N && A_col < N)
        {
            shared_A[threadIdx.y][threadIdx.x] =
                A[row * N + A_col];
        }
        else
        {
            shared_A[threadIdx.y][threadIdx.x] =
                0.0f;
        }

        // Cooperatively load B tile
        if (B_row < N && col < N)
        {
            shared_B[threadIdx.y][threadIdx.x] =
                B[B_row * N + col];
        }
        else
        {
            shared_B[threadIdx.y][threadIdx.x] =
                0.0f;
        }

        __syncthreads();

        // Reuse values from shared memory
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            sum +=
                shared_A[threadIdx.y][k] *
                shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}

/*
 * Return the median of the recorded runtimes.
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

/*
 * Benchmark naive GEMM.
 */
float benchmark_naive(
    const float* dA,
    const float* dB,
    float* dC,
    int N)
{
    dim3 block(
        BLOCK_SIZE,
        BLOCK_SIZE
    );

    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // Warmup
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

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> runtimes;
    runtimes.reserve(MEASUREMENTS);

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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return median(runtimes);
}

/*
 * Benchmark tiled GEMM.
 */
float benchmark_tiled(
    const float* dA,
    const float* dB,
    float* dC,
    int N)
{
    dim3 block(
        BLOCK_SIZE,
        BLOCK_SIZE
    );

    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // Warmup
    for (int run = 0; run < WARMUP_RUNS; run++)
    {
        tiled_gemm<<<grid, block>>>(
            dA,
            dB,
            dC,
            N
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> runtimes;
    runtimes.reserve(MEASUREMENTS);

    for (int run = 0; run < MEASUREMENTS; run++)
    {
        CUDA_CHECK(cudaEventRecord(start));

        tiled_gemm<<<grid, block>>>(
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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return median(runtimes);
}

/*
 * Calculate GEMM throughput.
 */
double calculate_gflops(
    int N,
    float milliseconds)
{
    double flops =
        2.0 *
        static_cast<double>(N) *
        N *
        N;

    double seconds =
        milliseconds / 1000.0;

    return flops / seconds / 1e9;
}

/*
 * Verify C for A = 1 and B = 2.
 */
bool verify_result(
    const std::vector<float>& C,
    int N)
{
    float expected =
        2.0f * N;

    for (size_t i = 0; i < C.size(); i++)
    {
        if (C[i] != expected)
        {
            std::cout
                << "Mismatch at index "
                << i
                << ": expected "
                << expected
                << ", got "
                << C[i]
                << "\n";

            return false;
        }
    }

    return true;
}

/*
 * Run one controlled naive-vs-tiled comparison.
 */
void run_experiment(int N)
{
    size_t num_elements =
        static_cast<size_t>(N) * N;

    size_t bytes =
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

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

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

    float naive_ms =
        benchmark_naive(
            dA,
            dB,
            dC,
            N
        );

    double naive_gflops =
        calculate_gflops(
            N,
            naive_ms
        );

    float tiled_ms =
        benchmark_tiled(
            dA,
            dB,
            dC,
            N
        );

    double tiled_gflops =
        calculate_gflops(
            N,
            tiled_ms
        );

    CUDA_CHECK(cudaMemcpy(
        C.data(),
        dC,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool passed =
        verify_result(
            C,
            N
        );

    double speedup =
        naive_ms / tiled_ms;

    std::cout
        << "N: "
        << N
        << "\n";

    std::cout
        << "Naive Median Time: "
        << naive_ms
        << " ms\n";

    std::cout
        << "Naive Performance: "
        << naive_gflops
        << " GFLOP/s\n";

    std::cout
        << "Tiled Median Time: "
        << tiled_ms
        << " ms\n";

    std::cout
        << "Tiled Performance: "
        << tiled_gflops
        << " GFLOP/s\n";

    std::cout
        << "Speedup: "
        << speedup
        << "x\n";

    std::cout
        << "Correctness: "
        << (passed ? "PASS" : "FAIL")
        << "\n";

    std::cout
        << "----------------------------------------"
        << "\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main()
{
    const std::vector<int> sizes = {
        1024,
        2048,
        4096,
        8192
    };

    std::cout
        << "Controlled Naive vs Tiled GEMM Scaling Experiment\n";

    std::cout
        << "Block configuration: "
        << BLOCK_SIZE
        << "x"
        << BLOCK_SIZE
        << " for both kernels\n";

    std::cout
        << "========================================\n";

    for (int N : sizes)
    {
        run_experiment(N);
    }

    return 0;
}