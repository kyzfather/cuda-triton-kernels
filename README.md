# CUDA/Triton GPU 算子实现集合

## 简介

本仓库记录了我在 [LeetGPU](https://leetgpu.com/) 平台上实现的一系列 GPU 算子，涵盖 elementwise、reduction、softmax、归一化、矩阵乘、Attention 等 LLM 推理中常见的核心算子。每个算子均实现了 Triton 版本（部分附 Cuda 版本），并针对性地应用了不同的 GPU 性能优化手段。部分算子使用ncu（nsight compute）进行了分析。

- 语言/框架：CUDA C++ / Triton
- 平台：LeetGPU / 云平台
- 个人主页：https://leetgpu.com/KYZ

---
## NCU分析调优，以及优化手段小结（Todo）



- **访存优化**：
- **计算优化**：
- **算子融合**：
- **Attention相关**：

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