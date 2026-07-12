#include <cuda_runtime.h>

__device__ float reduce(float val) {
    for (int stride = 16; stride >= 1; stride /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    return val;
}

// 这里的实现相比triton版本实现的问题，就是input数据load了两次，而triton里是用寄存器抗的
// 需要把读取的input放到smem或者寄存器里，其次是向量化加载
__global__ void rmsnorm(const float* input, float gamma, float beta, float* output, int N,
                        float eps) {
    // 写了遍triton 再写遍cuda，就是block规约
    __shared__ float smem[32];
    int pid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int warp_count = blockDim.x / 32;
    float sum = 0.0f;
    for (int i = pid; i < N; i += stride) {
        sum += input[i] * input[i];
    }

    // warp reduce
    sum = reduce(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();  // sync for warps

    if (warp_id == 0) {
        float val = (lane_id >= warp_count) ? 0.0 : smem[lane_id];
        val = reduce(val);
        if (lane_id == 0) {
            smem[0] = val;
        }
    }

    __syncthreads();

    float rms = sqrt(smem[0] / N + eps);
    for (int i = pid; i < N; i+= stride) {
        output[i] = input[i] * gamma / rms + beta;
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    int block_size = 256;
    int grid = 1;
    rmsnorm<<<grid, block_size>>>(input, gamma, beta, output, N, eps);                    
}
