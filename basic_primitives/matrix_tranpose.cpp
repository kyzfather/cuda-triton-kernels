#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;  
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float smem[16][16 + 1];
    if (ix < cols && iy < rows) {
        smem[threadIdx.y][threadIdx.x] = input[iy * cols + ix];
    }

    __syncthreads();

    // 这里需要保证block内的线程在访问output的时候是合并内存访问的
    // 原来访问的是input的（blockIdx.y, blockIdx.x）的那个块
    // 转置后对应的是output的(blockIdx.x, blockIdx.y)的那个块
    int ox = blockDim.y * blockIdx.y + threadIdx.x;
    int oy = blockDim.x * blockIdx.x + threadIdx.y;
    output[oy * rows + ox] = smem[threadIdx.x][threadIdx.y];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
