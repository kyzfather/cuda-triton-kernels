# CUDA/Triton GPU 算子实现集合

## 简介

本仓库记录了我在 [LeetGPU](https://leetgpu.com/) 平台上实现的一系列 GPU 算子，涵盖 elementwise、reduction、softmax、归一化、矩阵乘、Attention 等 LLM 推理中常见的核心算子。每个算子均实现了 CUDA 版本（部分附 Triton 版本），并针对性地应用了不同的 GPU 性能优化手段。

- 语言/框架：CUDA C++ / Triton
- 平台：LeetGPU / 本地实测（如有）
- 个人主页：https://leetgpu.com/KYZ

---

## 算子目录

### 1. Elementwise / 基础算子
| 算子 | 优化点 | 耗时/带宽利用率 | 平台排名（如有） |
|---|---|---|---|
| Vector Add | 向量化访存 (float4) | xx GB/s | Top x% |
| ... | ... | ... | ... |

### 2. Reduction 类
| 算子 | 优化点 | 耗时 | 备注 |
|---|---|---|---|
| Sum/Max Reduction | warp shuffle reduce + shared memory | xx us | 相比naive提速 xx倍 |
| Softmax | online softmax，避免两次遍历 | xx us | |

### 3. 归一化类
| 算子 | 优化点 | 耗时 |
|---|---|---|
| LayerNorm | 一次遍历求均值方差(Welford)，warp reduce | xx us |
| RMSNorm | 融合element-wise激活函数 | xx us |

### 4. 矩阵乘 / GEMM
| 算子 | 优化点 | 性能 vs cuBLAS |
|---|---|---|
| Naive GEMM → Tiled GEMM | shared memory tiling，避免bank conflict | 达到cuBLAS xx% |

### 5. Attention
| 算子 | 优化点 | 备注 |
|---|---|---|
| FlashAttention (Triton) | tiling + online softmax，减少显存读写 | 复现核心逻辑，对比naive attention显存占用降低xx% |
| FlashDecoding (Triton) | 长上下文场景的并行拆分 | |

> 注：以上数据为示例格式，请替换为你的真实测试数据/截图/排名。

---

## 优化手段小结

按你实际用到的技术点分类总结，方便面试时按图索骥讲清楚，例如：

- **访存优化**：向量化访存（float4/float2）、合并访存（coalesced access）、shared memory 复用
- **计算优化**：warp-level primitives（shfl_down等）、减少bank conflict、增大计算密度
- **算子融合**：xxx与xxx融合，减少kernel launch开销和中间结果显存读写（可以关联你在拼多多的融合经验）
- **Attention相关**：tiling、online softmax、causal mask处理、long context场景的显存优化

---

## 目录结构

```
.
├── elementwise/
├── reduction/
├── normalization/
├── gemm/
├── attention/
└── README.md
```

---

## 后续计划（可选）

- [ ] 补充更多量化相关算子（INT8/FP8）
- [ ] 补充投机推理相关的验证/采样算子
- [ ] 增加与PyTorch/cuBLAS的系统性benchmark对比脚本