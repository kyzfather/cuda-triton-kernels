#include <cuda_runtime.h>

__global__ scan(const float* data, float* block_sums, int N) {
    // 我是感觉每个block处理的元素数目是，block的线程数的两倍
    // 先每个block进行局部的prefix sum, exclusive
    // 最后的block的规约该怎么做呢？每个线程负责原本一个block的规约？
    __shared__ float smem[512];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = blockIdx.x * blockDim.x;
    int end = start + blockDim.x;

    if (tid < N) {
        smem[threadIdx.x] = data[tid];
    }

    // 向上
    for (int stride = 2; stride < blockDim.x; stride *= 2) {
        int idx = start + (tid + 1) * stride - 1;
        if (idx < end && idx < N) {
            smem[idx] += smem[idx - stride / 2];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = smem[blockDim.x - 1];
        }
        smem[blockDim.x - 1] = 0.0f;
    }

    // 向下
    for (int stride = blockDim.x; stride >= 2; stride /= 2) {
        int idx = start + (tid + 1) * stride - 1;

        if (idx < end && idx < N) {
            // swap(left, right)
            float tmp = smem[idx - stride / 2];
            smem[idx - stride / 2] = smem[idx];
            smem[idx] = tmp;
            smem[idx] += smem[idx - stride / 2];
        }
        __syncthreads();
    }

    if (tid < N) {
        data[tid] = smem[threadIdx.x];
    }
}

__global__ void add(float* output, float* block_sums, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] == block_sums[blockIdx.x];
    }
}

void prefix_sum(const float* data, int N) {
    int BLOCK_SIZE = 256;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 直到能单个block算出结果
    if (GRID_SIZE == 1) {
        scan<<<GRID_SIZE, BLOCK_SIZE>>>(input, nullptr, output, N);
        return;
    }

    float* d_block_sums;
    cudaMalloc(&d_block_sums, GRID_SIZE * sizeof(float));

    scan<<<GRID_SIZE, BLOCK_SIZE>>>(data, d_block_sums, N);

    // block_sums的prefix sum需要算出来，这里递归
    prefix_sum(d_block_sums, N);

    add<<<GRID_SIZE, BLOCK_SIZE>>>(data, d_block_sums, N);
    cudaFree(d_block_sums);
}
