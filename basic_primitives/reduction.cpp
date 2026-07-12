#include <cuda_runtime.h>
#include <cstdint>

static const int WARP_COUNT = 8;
__device__ float warp_reduce(float val) {
    for (int stride = 16 ; stride >= 1; stride /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, stride);
    }
    return val;
}
__global__ void reduce(const float* input, float* output, int N) {
    __shared__ float smem[WARP_COUNT];  // warp count must be smaller than 33
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t thread_count = gridDim.x * blockDim.x;
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    float data = 0;

    for (; tid < N; tid += thread_count) {
        data += input[tid];
    }

    float val = warp_reduce(data);
    if (lane_id == 0) {
        smem[warp_id] = val;
    }

    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < WARP_COUNT ? smem[lane_id] : 0;
        val = warp_reduce(val);
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, val);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    cudaMemset(output, 0, sizeof(float));
    int block_size = WARP_COUNT * 32;
    int grid_size = (N + block_size - 1) / block_size;
    reduce<<<grid_size, block_size>>>(input, output, N);
}

