# CUDA/Triton GPU 算子实现集合

## 简介

本仓库记录了我在 [LeetGPU](https://leetgpu.com/) 平台上实现的一系列 GPU 算子，涵盖 elementwise、reduction、softmax、归一化、矩阵乘、Attention 等 LLM 推理中常见的核心算子。每个算子均实现了 Triton 版本（部分附 Cuda 版本），并针对性地应用了不同的 GPU 性能优化手段。部分算子使用ncu（nsight compute）进行了分析。

- 语言/框架：CUDA C++ / Triton
- 平台：LeetGPU / 云平台
- 个人主页：https://leetgpu.com/KYZ

---
## 优化手段与实现小结

### Activation

- **Softmax**：采用 online safe-softmax。naive 实现需要三趟遍历（找最大值 → 求和 → 计算结果），online-softmax 通过递推将"找最大值"和"求和"合并为一趟。思路上类似 LayerNorm 用 Welford 算法将"求均值"和"求方差"的两趟遍历合并为一趟。若特征维度不大、shared memory 消耗有限，也可以直接一次性 load 到 smem 中做规约，不需要递推。

- **SwiGLU**：通过 kernel 融合减少中间结果的访存次数。
  - kernel 0：`x` 与 `cat(w_gate, w_up)` 做矩阵乘
  - kernel 1（融合层）：计算 `x · sigmoid(x) · y`
  - kernel 2：与 `w_down` 做矩阵乘

### Attention

- **Sliding Window Attention**：在 kv_seq_len 维度遍历时，每个 query 只需访问其对应的滑动窗口范围，而非整个序列，减少了不必要的计算和访存开销。

- **INT8 KV-Cache Attention**:  Decode 阶段的 MHA（此时 Query 序列长度为 1）。Decode 阶段主要是 memory-bound, 通过将 KV Cache 进行 INT8 量化以大幅减少数据搬运量，并在计算时反量化为 FP16。由于单 Query 无法在 Sequence 维度有效展开并行，本算子在 kv_seq_len 维度引入跨 Block 并行，最后通过一个独立的 Reduce Kernel 对该维度进行最终规约。

### Elementwise

- **Vector Addition**： 
  - 使用 `float4` 向量化访存，减少发射到 LSU（Load/Store Unit）的指令数量
  - 采用 stride loop（每个线程跨步处理多组数据），在 warp 数量已经很多的情况下，通过增加单个 warp 的任务量而非启动更多 warp，更利于指令间的延迟隐藏（latency hiding）

### Basic Primitives

- **Matrix Transpose**：借助 shared memory 作为中转媒介，解决 transpose 场景下 load 和 store 无法同时满足合并访存（coalesced access）的问题。

- **Reduction（Sum / Mean / Max）**：每个 block 负责规约一组数据（如一行），每个线程可跨步读取多个元素。采用两级规约策略：
  1. **Warp 内规约**：通过 `__shfl_xor_sync` 完成 warp 内部的规约
  2. **Warp 间规约**：每个 warp 的 0 号线程将规约结果写入 shared memory 对应位置，最后由 warp 0 对这些中间结果做最终的跨 warp 规约

- **Prefix Sum**：
  - **单 block 场景**：采用类似二叉树结构的两趟扫描——向上扫描（up-sweep）完成求和，向下扫描（down-sweep）完成数据交换
  - **多 block 场景**：每个 block 先计算自身的局部前缀和，同时统计各自的 block sum 并按序写入显存；随后递归地对这些 block sum 求前缀和；最后将对应的前缀结果累加回每个 block 内部，得到全局前缀和

### GEMM

- **GEMM（Triton 实现）**：采用分块（tiling）策略，核心在于合理调节 `Br`（行块大小）和 `Bc`（列块大小），计算密度近似为：`I ≈ (Br · Bc) / (Br + Bc)`

  - `Br`、`Bc` 过小 → 计算密度不足，退化为 memory-bound，容易出现 long scoreboard stall
  - `Br`、`Bc` 过大 → shared memory 消耗增加，降低 occupancy，进而减少 scheduler 可调度的 warp 数量

  需要在计算密度与 occupancy 之间找到平衡点。

### MOE
- **MoE Top-K Gating**：M 个 token， E 个专家，每个 block 处理一个token， 循环 k 次 reduce 求最大值，找到 top-k 的 logits 值，然后计算 softmax 值。（因为专家的数量不会太大， k 也不会太大，所以通过这种 k 次 reduce 的方式）

### Sampling & Decoding

- **Top-K Selection**：
  - **n 较大时**（如词表规模）：每个 block 负责一部分数据，内部使用双调排序（bitonic sort）找出局部 top-k，再将各 block 的局部结果写入 global memory，递归处理直至单个 block 可以完成最终归并
  - **n 较小时**（如专家数量场景，k 也较小）：直接进行 k 次 reduce 找最大值即可，无需完整排序，开销更低

---
## Nsight Compute Profiling 小结

### GPU Speed of Light

这一页主要看两个维度：**计算资源利用率**和**带宽利用率**，并结合 **Roofline 模型**判断kernel的瓶颈类型。

**Roofline 模型核心概念：**

- **计算密度（Arithmetic Intensity）** = 总计算量（FLOPs）/ 总访存量（Bytes），横轴（X轴），由具体算法/kernel的实现决定
- **纵轴（Y轴）**：理论可达到的性能上限（FLOPs/s）
- **斜率**：硬件的最大带宽，由硬件规格决定，与算法无关
- **拐点（Ridge Point）**：由硬件的"最大计算能力 / 最大带宽"算出的计算密度阈值
  - 若算法的计算密度落在拐点**左侧** → **memory-bound**
  - 若落在拐点**右侧** → **compute-bound**

在 ncu 实际的 Roofline 图中，坐标轴取了对数，Y轴的截距对应带宽大小，图上通常会同时画出 **L1 Cache、L2 Cache、DRAM** 三条不同带宽的屋顶线，可以据此判断当前kernel更接近哪一级存储的瓶颈。

---

### Warp State Statistics

Warp Stall 的几种常见类型及对应的优化方向：

| Stall 类型 | 含义 | 常见诱因 | 优化方向 |
|---|---|---|---|
| **Long Scoreboard** | 等待 global memory 返回数据，访存延迟未被掩盖 | memory-bound kernel 的典型表现 | 提高计算密度以减少相对访存量、让数据搬运与计算重叠（如 double buffering / async copy）、合并内存访问（coalesced access） |
| **Math Pipe Throttle** | 计算单元（FMA/ALU/SFU/Tensor Core等）本身发射队列已满 | kernel 是 compute-bound | 若同时观察到其他硬件单元空闲，可能是 scheduler 可调度的 warp 数量太少、延迟未被充分隐藏，可考虑提升 occupancy |
| **MIO Throttle** | MIO（Memory I/O，包含 shared memory、texture、部分特殊指令）流水线发射队列已满 | shared memory 访问过于频繁、存在 bank conflict、大量小粒度的 shared memory load/store | 合并 shared memory 访问、消除 bank conflict、减少访问次数 |
| **Stall Wait** | 等待固定延迟依赖完成（如某些数学指令的 pipeline depth） | 指令级并行（ILP）不足 | 增加单线程处理的数据量、重排指令顺序以提升 ILP |
| **Stall Barrier** | Block 内各 warp 执行进度不一致，同步点（如 `__syncthreads()`、`cp.async.wait`）处快的 warp 需等待最慢的 warp | 同步开销、warp 间负载不均衡 | 尽量减少同步点、平衡各 warp 的工作量 |
| **Short Scoreboard** | 与 Long Scoreboard 类似，但等待的是片上资源（shared memory、texture 等延迟较短的 MIO 指令结果） | shared memory 的 load/store 延迟未被掩盖、bank conflict（通常会同时拉高 MIO Throttle） | 优化 shared memory 访问模式、消除 bank conflict |

> **补充说明（7.14）**：MIO Throttle 与 shared memory 以及 SFU（`exp`、`log`、`sin`、`cos` 等特殊函数单元）密切相关——当这两类 pipeline 的指令队列被打满时就会触发该 stall。
---

### Occupancy

每个 SM 上的 shared memory 和寄存器（register）资源有限，因此可容纳的最大 warp 数 / block 数存在理论上限。Kernel 实际消耗的 shared memory 和寄存器数量，决定了每个 SM 实际能容纳的 block 数量。

Occupancy 应尽量维持在较高水平，这样每个 warp scheduler 才有足够多的可调度 warp，从而更好地隐藏访存/计算延迟。

> Occupancy 不是越高越好，关键在于是否"足够"隐藏延迟——如果 kernel 本身 ILP 较高、单 warp 就能有效隐藏延迟，过度追求 occupancy 反而可能挤压每个线程可用的寄存器/shared memory 资源。

---

### Instruction Statistics

分析 kernel 执行过程中各类指令的数量占比。

- **FP32 Non-Fused Instructions**：如果代码中的乘法和加法可以合并（如 `a * b + c`），编译器/手写代码应尽量利用 **FMA（Fused Multiply-Add）** 指令，将两条指令合并为一条，减少总指令发射数量，提升吞吐。
---

## 目录结构

```
.
├── activation/
├── attention/
├── basic_primitives/
├── convolution/
├── elementwise/
├── gemm/
├── moe/
├── normalization/
├── quantization/
├── sampling_decoding/
└── README.md
```

---

## 后续计划（可选）

- [ ] 补充更多量化相关算子（INT8/FP8）
- [ ] 补充投机推理相关的验证/采样算子