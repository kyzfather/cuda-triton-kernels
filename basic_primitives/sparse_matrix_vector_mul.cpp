#include <cuda_runtime.h>

__device__ float warp_reduce(float val) {
    for (int stride = 16; stride >= 1; stride /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    return val;
}

__global__ void sparse_matrix_vector_mul(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int row = blockIdx.x;
    int stride = blockDim.x;
    int offsets = row * N;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    // 感觉也是规约， block内每个线程计算自己的部分，然后warp内规约
    // 然后block内的规约 但是这里的sparse不知道怎么用上
    float val = 0.0f;
    for (int id = threadIdx.x; id < N; id += stride) {
        val += A[offsets + id] * x[id];
    }

    // warp reduce
    val = warp_reduce(val);
    // int SMEM_SIZE = blockDim.x / 32;
    int SMEM_SIZE = 8;
    __shared__ float smem[8];
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // block reduce
    if (warp_id == 0) {
        float tmp = (lane_id < SMEM_SIZE ? smem[lane_id] : 0.0f);
        float res = warp_reduce(tmp);
        if (threadIdx.x == 0) {
            y[blockIdx.x] = res;
        }
    }
}


// compressed sparse row(CSR)的格式实现一下
// A = [[1, 0, 2],
//      [0, 3, 0],
//      [4, 0, 5]]

// values     = [1, 2, 3, 4, 5]        # 非零值
// col_indices = [0, 2, 1, 0, 2]       # 每个非零值的列索引
// row_ptrs   = [0, 2, 3, 5]           # 每行非零值的起始位置（长度 M+1）

// 每个线程处理一行。 缺点：每行的元素数目不同，造成warp divergency
// 还有一种实现：每个warp处理一行，适合列数较大，不会造成warp divergency
__global__ void sparse_matrix_vector_mul_csr(
    const int M,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices, 
    const float* __restrict__ value,
    const float** __restrict__ x,
    float* __restrict__ y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        float sum = 0.0f;
        for (int i = row_start; i < row_end; i++) {
            int col = col_indices[i];
            sum += values[i] * x[col];
        }

        y[row] = sum;
    }
}


// 太复杂了，还要prefix scan
// 就假设给的矩阵就是csr格式吧
__global__ convert_dense_to_sparse_csr(const float* input, float* val, int* indices, int* row_ptrs, 
                                      int M, int N, int nnz) {
    //                                    
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int block_size = 256;
    int grid_size = M;
    sparse_matrix_vector_mul<<<grid_size, block_size>>>(A, x, y, M, N, nnz);
}
