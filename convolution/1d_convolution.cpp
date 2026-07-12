#include <cuda_runtime.h>
#include <stdio.h>

__global__ void convolution_1d_kernel(const float* __restrict__ input,
                                      const float* __restrict__ kernel,
                                      float* output,
                                      int input_size,
                                      int kernel_size)
{
    extern __shared__ float smem[];
    float* s_input = smem;                 // 输入数据 shared memory
    float* s_kernel = smem + blockDim.x + kernel_size - 1;  // kernel shared memory

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // 每个线程加载 kernel 到 shared memory
    for(int i = tx; i < kernel_size; i += blockDim.x) {
        s_kernel[i] = kernel[i];
    }

    // 每个线程加载自己的输入元素
    if (tid < input_size)
        s_input[tx] = input[tid];
    else
        s_input[tx] = 0.0f;

    // 加载 halo 区域
    for (int offset = tx; offset < kernel_size - 1; offset += blockDim.x) {
        int g_idx = blockIdx.x * blockDim.x + blockDim.x + offset;
        s_input[blockDim.x + offset] = (g_idx < input_size) ? input[g_idx] : 0.0f;
    }

    __syncthreads();

    // 只计算有效输出
    int output_size = input_size - kernel_size + 1;
    if (tid < output_size) {
        float tmp = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            tmp += s_input[tx + i] * s_kernel[i];
        }
        output[tid] = tmp;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size)
{
    int output_size = input_size - kernel_size + 1;
    if (output_size <= 0) return;

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    // shared memory = 输入段 + kernel
    size_t smem_size = (threadsPerBlock + kernel_size - 1 + kernel_size) * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(
        input, kernel, output, input_size, kernel_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("CUDA runtime error: %s\n", cudaGetErrorString(err));
}