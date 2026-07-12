#include <cuda_runtime.h>

// welford算法的实现
// 就是一遍算出均值mean和方差var, 递推公式，由前一项推出当前项
// m_n = m_n-1 + (x_n - m_n-1) / n
// s_n = s_n-1 + (x_n - m_n-1)(x_n - m_n)

// 分块的递推
// m_x = m_a + (m_b - m_a) * n_b / (n_a + n_b)
// s_x = s_a + s_b + (m_b - m_a)^2 * n_a * n_b  / (n_a + n_b)


// 注意，结构体默认值不是0，是随机值
// 需要{}初始化，才为0，例如WelfordData data{};
struct WelfordData {
    float mean;
    float s;
    int count;
};

// 这里用块的递推
__device__ WelfordData combine(WelfordData a, WelfordData b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;

    WelfordData out{};

    int n = a.count + b.count;
    float delta = b.mean - a.mean;

    out.mean = a.mean + delta * b.count / n;
    out.s = a.s + b.s + delta * delta * a.count * b.count / n;
    out.count = n;

    return out;
}

// 注意，不要用__shlf_xor_sync交换结构体或者类
// (乱想：学习和上班一点都不一样，上班上上就累了，因为是给别人上的，但学习是提升自己的，
//  所以还是学习能坚持的时间更久一些)
__device__ WelfordData reduce(WelfordData val) {
    for (int stride = 16; stride > 0; stride >>= 1) {
        WelfordData tmp{};
        tmp.count = __shfl_xor_sync(0xffffffff, val.count, stride);
        tmp.mean = __shfl_xor_sync(0xffffffff, val.mean, stride);
        tmp.s = __shfl_xor_sync(0xffffffff, val.s, stride);
        val = combine(tmp, val);
    }
    return val;
}


__global__ void layernorm(const float* input, float* output, int N, const float* gamma, const float* beta, float eps) {
    __shared__ WelfordData smem[32];

    int row = blockIdx.x;   
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int warp_cnt = blockDim.x / 32;
    const float* start = input + row * N;
    WelfordData tmp{};
    // m_n = m_n-1 + (x_n - m_n-1) / n
    // s_n = s_n-1 + (x_n - m_n-1)(x_n - m_n)
    for (int i = threadIdx.x; i < N; i+= blockDim.x) {
        float val = start[i];
        float delta0 = val - tmp.mean;
        tmp.count += 1;
        tmp.mean = tmp.mean + delta0 / tmp.count;
        tmp.s = tmp.s + delta0 * (val - tmp.mean);
    }

    // warp reduce
    tmp = reduce(tmp);
    if (lane_id == 0) {
        smem[warp_id] = tmp;
    }

    // sync: wait all warps done
    __syncthreads();


    // block reduce: warp0 do this reduce
    if (warp_id == 0) {
        tmp = (lane_id >= warp_cnt) ? WelfordData{} : smem[lane_id];
        tmp = reduce(tmp);

        if (lane_id == 0) {
            smem[0] = tmp;
        }
    }

    __syncthreads();

    tmp = smem[0];

    float rsqrt_var = rsqrtf(tmp.s / tmp.count + eps);

    for (int id = threadIdx.x; id < N; id += blockDim.x) {
        float gamma_val = gamma[id];
        float beta_val = beta[id];

        output[row * N + id] = gamma_val * (start[id] - tmp.mean) * rsqrt_var + beta_val;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N,
                      float eps) {
    int block_size = 256;
    int grid = M;
    layernorm<<<grid, block_size>>>(input, output, N, gamma, beta, eps);                    
}