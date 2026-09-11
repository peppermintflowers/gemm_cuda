#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
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

constexpr int N = 4096;
constexpr int BLOCK_SIZE = 16;
constexpr int WARMUP_RUNS = 3;

/*
 * Naive GEMM.
 * One thread computes one element of C.
 * A and B operands are accessed through global memory.
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
 * Uses the same thread-block configuration and output mapping
 * as naive GEMM, but explicitly stages A and B tiles in
 * shared memory before reusing them.
 */
__global__ void tiled_gemm(
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

        // Cooperatively stage one A tile
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

        // Cooperatively stage one B tile
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

        // Reuse staged values for this K tile
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
 * Verify C for A = 1 and B = 2.
 */
bool verify_result(
    const std::vector<float>& C)
{
    const float expected =
        2.0f * N;

    for (size_t i = 0; i < C.size(); i++)
    {
        if (C[i] != expected)
        {
            std::cerr
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

int main()
{
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

    dim3 block(
        BLOCK_SIZE,
        BLOCK_SIZE
    );

    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    /*
     * Warm up both kernels before profiling.
     * These launches are outside the profiler capture range.
     */
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

        tiled_gemm<<<grid, block>>>(
            dA,
            dB,
            dC,
            N
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /*
     * Only the following two kernel launches are exposed
     * to Nsight Compute for profiling.
     */
    CUDA_CHECK(cudaProfilerStart());

    naive_gemm<<<grid, block>>>(
        dA,
        dB,
        dC,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    tiled_gemm<<<grid, block>>>(
        dA,
        dB,
        dC,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStop());

    // Verify the final tiled result
    CUDA_CHECK(cudaMemcpy(
        C.data(),
        dC,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool passed =
        verify_result(C);

    std::cout
        << "N: "
        << N
        << "\n";

    std::cout
        << "Block configuration: "
        << BLOCK_SIZE
        << "x"
        << BLOCK_SIZE
        << "\n";

    std::cout
        << "Correctness: "
        << (passed ? "PASS" : "FAIL")
        << "\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return passed ? 0 : 1;
}