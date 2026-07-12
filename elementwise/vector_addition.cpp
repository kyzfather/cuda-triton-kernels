#include <cuda_runtime.h>

// naive solution
// better solution: float4 + grid stide loop
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n4 = N / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);
    for (int idx = tid; idx < n4; idx += stride) {
        float4 a = A4[idx];
        float4 b = B4[idx];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[idx] = c;
    }

    int start = n4 * 4;
    int idx = start + tid;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int blockSize = 256;
    int processPerBlock = blockSize * 4;
    int deviceId = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);
    int smCount = prop.multiProcessorCount;
    int maxBlockSize = smCount * 4;
    int gridSize = (N + processPerBlock - 1) / processPerBlock;
    gridSize = min(gridSize, maxBlockSize);

    vector_add<<<gridSize, blockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
