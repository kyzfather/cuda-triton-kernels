#include <cuda_runtime.h>

// 多block的实现 exclusive
// 1. 每个block计算自己区间的prefix sum
// 2. block0来统计每个block的sum, 搞成一个数组，然后计算这个的前缀和。然后数组往右偏移一位
// 3. 最后每个block读取这个数组的对应位置，将自己区间的prefix sum加上对应的值
__global__ void scan_blocks(float* data, float* block_sums, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? data[gid] : 0;
    __syncthreads();

    int block_n = blockDim.x;

    // up
    for (int stride = 1; stride < block_n; stride *= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < block_n) {
            smem[idx] += smem[idx - stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = smem[block_n - 1];
        }
        smem[block_n - 1] = 0.0f;
    }
    __syncthreads();

    // down
    for (int stride = block_n / 2; stride >= 1; stride /= 2) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < block_n) {
            float tmp = smem[idx - stride];
            smem[idx - stride] = smem[idx];
            smem[idx] += tmp;      
        } 
        __syncthreads();
    }

    if (gid < n) {
        data[gid] = smem[tid];
    }
}

__global__ void exclusive_to_inclusive(const float* input, float* sum, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int id = tid; id < n; id += stride) {
        sum[id] += input[id];
    }
}

__global__ void add_block_sums(float* data, float* block_sums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        data[gid] += block_sums[blockIdx.x];
    }
}

void parallel_prefix_sum(float* data, int n) {
    int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int smem_size = BLOCK_SIZE * sizeof(float);

    if (num_blocks <= 1) {
        scan_blocks<<<1, BLOCK_SIZE, smem_size>>>(data, nullptr, n);
        return;
    }

    float* block_sums;
    cudaMalloc(&block_sums, num_blocks * sizeof(float));

    // step1
    scan_blocks<<<num_blocks, BLOCK_SIZE, smem_size>>>(data, block_sums, n);

    // step2: 对block_sums做scan(递归)
    parallel_prefix_sum(block_sums, num_blocks);

    // step3
    add_block_sums<<<num_blocks, BLOCK_SIZE>>>(data, block_sums, n);
    cudaFree(block_sums);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    cudaMemcpy(output, input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    parallel_prefix_sum(output, N);
    int block_size = 256;
    int grid = (N + block_size - 1) / block_size;
    exclusive_to_inclusive<<<grid, block_size>>>(input, output, N);
}