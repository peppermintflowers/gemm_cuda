#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

/* Checks for and prints error encountered */
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
 * Tiled computation for C = A x B
 */
template <int TILE_SIZE>
__global__ void tiled_gemm(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    // Global column index for thread
    int col =
        blockIdx.x * TILE_SIZE +
        threadIdx.x;

    // Global row index for thread
    int row =
        blockIdx.y * TILE_SIZE +
        threadIdx.y;

    // Shared tiles for A and B
    __shared__ float shared_A_data[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B_data[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Iterate through tiles along the shared dimension
    for (int tile = 0; tile < N; tile += TILE_SIZE)
    {
        // Global column of A needed for this tile
        int A_col =
            tile + threadIdx.x;

        // Global row of B needed for this tile
        int B_row =
            tile + threadIdx.y;

        // Each thread loads one A value into shared memory
        if (row < N && A_col < N)
        {
            shared_A_data[threadIdx.y][threadIdx.x] =
                A[row * N + A_col];
        }
        else
        {
            // Pad out-of-bounds values with zero
            shared_A_data[threadIdx.y][threadIdx.x] =
                0.0f;
        }

        // Each thread loads one B value into shared memory
        if (B_row < N && col < N)
        {
            shared_B_data[threadIdx.y][threadIdx.x] =
                B[B_row * N + col];
        }
        else
        {
            // Pad out-of-bounds values with zero
            shared_B_data[threadIdx.y][threadIdx.x] =
                0.0f;
        }

        // Wait until every thread has finished loading the tile
        __syncthreads();

        // Compute this tile's contribution to C[row,col]
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum +=
                shared_A_data[threadIdx.y][k] *
                shared_B_data[k][threadIdx.x];
        }

        /* Wait until every thread has finished using the tile
        before shared memory is overwritten with the next tile */
        __syncthreads();
    }

    // Store final result in global memory
    if (row < N && col < N)
    {
        C[row * N + col] = sum;
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

/*
 * Benchmark tiled GEMM for one tile size
 */
template <int TILE_SIZE>
void benchmark_tiled_gemm(
    const float* dA,
    const float* dB,
    float* dC,
    int N)
{
    dim3 block(
        TILE_SIZE,
        TILE_SIZE
    );

    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    // Warmup runs
    for (int run = 0; run < WARMUP_RUNS; run++)
    {
        tiled_gemm<TILE_SIZE><<<grid, block>>>(
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

    // Measure kernel execution time
    for (int run = 0; run < MEASUREMENTS; run++)
    {
        CUDA_CHECK(cudaEventRecord(start));

        tiled_gemm<TILE_SIZE><<<grid, block>>>(
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
        << "Tile Size: "
        << TILE_SIZE
        << "x"
        << TILE_SIZE
        << "\n";

    std::cout
        << "Threads/Block: "
        << TILE_SIZE * TILE_SIZE
        << "\n";

    std::cout
        << "Median Time: "
        << median_ms
        << " ms\n";

    std::cout
        << "Performance: "
        << gflops
        << " GFLOP/s\n\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

/*
 * Check GEMM output for A = 1.0 and B = 2.0
 */
bool verify_result(
    const std::vector<float>& C,
    int N)
{
    const float expected =
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

    std::cout
        << "N: "
        << N
        << "\n\n";

    // Benchmark different tile sizes
    benchmark_tiled_gemm<8>(
        dA,
        dB,
        dC,
        N
    );

    benchmark_tiled_gemm<16>(
        dA,
        dB,
        dC,
        N
    );

    benchmark_tiled_gemm<32>(
        dA,
        dB,
        dC,
        N
    );

    // Copy final result back to CPU to verify correctness
    CUDA_CHECK(cudaMemcpy(
        C.data(),
        dC,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool passed =
        verify_result(C, N);

    std::cout
        << "Correctness: "
        << (passed ? "PASS" : "FAIL")
        << "\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}