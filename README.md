# Cinder

**Cinder** 是一个基于 MLIR 的 AI 推理计算图编译器内核。它演示了如何自定义一个高层张量方言，并将其逐步降级到 LLVM IR——对于 GPU 目标，则一路降级到 NVVM/PTX——同时对比不同降级策略的性能收益。

项目聚焦三个具体目标：

1. **定义高层张量方言（FCG）** —— 将 AI 推理计算图表达为可验证的 MLIR IR。
2. **构建完整降级管道** —— FCG → Linalg → SCF → LLVM（GPU 目标则为 FCG → NVVM），每一步都可观察、可测试。
3. **提供可量化的性能对比** —— 通过统一的基准测试矩阵，衡量每种优化策略的实际收益。

---

## FCG（Fast Computation Graph）

FCG 是 Cinder 内核中的高层动态形状张量方言。它刻意保持精简——仅支持二维动态形状——并以确定性的方式降级到 LLVM。

| 算子 | 说明 |
| :--- | :--- |
| `fcg.matmul` | 矩阵乘法 |
| `fcg.add` | 逐元素加法 |
| `fcg.relu` | ReLU 激活 |
| `fcg.fullflow` | 融合的 `matmul + add + relu` |

方言遵循 destination-passing style（DPS）约定：每个算子都接收一个显式的 `out` 张量并返回同一张量，从而在降级过程中消除分配压力。算子携带 `assemblyFormat`、自定义 verifier，以及供 NVVM 降级使用的 GPU 内核名访问器（`getKernelNameF32/F16`）。

---

## 降级管道

同一份 FCG IR 通过 **五种** 策略进行降级，各自以不同方式在抽象程度与性能之间做权衡。

### `fcg-pass-baseline`
将 `fcg.matmul` 降级为基于 `scf.for` + `memref` + `arith` 的朴素三重循环，纯标量代码，作为 1.0x 性能基准。

### `fcg-pass-vectorize`
在最内层循环引入 SIMD，通过 `vector.transfer_read` / `transfer_write` 一次性处理 32 个元素的列块。

### `fcg-pass-linalg`
不再手写循环，而是降级到 `linalg.generic` / `linalg.matmul`，交由 MLIR 官方优化管道（循环分块、融合、向量化）生成高性能代码。

### `fcg-pass-gpu`
通过 `gpu.launch` / `gpu.thread_id` 将计算映射到 GPU 线程块，生成可后续用 `convert-gpu-to-nvvm` 处理的 `gpu.module` + `gpu.func`。

### `fcg-pass-nvvm-gpu` / `fcg-pass-nvvm-cpu`
完全绕过 `gpu` 方言，直接降级到 NVVM + LLVM。GPU 侧 pass 将模块重写为纯 `nvvm.kernel` 函数，其参数为扁平化的 `!llvm.ptr<1>` 全局指针加整数维度（不含 `memref`）；CPU 侧 pass 则：

- 将编译好的 CUBIN 以字节数组的形式嵌入到全局符号中，
- 为每个 FCG 算子生成宿主端包装函数，负责从 `memref` 中提取设备指针与维度，
- 生成 CUDA Driver API 运行时（`cuInit`、`cuCtxCreate`、`cuModuleLoadData`、`cuLaunchKernel`），并通过 `llvm.mlir.global_ctors` / `global_dtors` 注册构造/析构函数。

由此得到一个完全自包含的动态库：运行时自行加载内嵌的 cubin 并通过 Driver API 启动内核，内核启动无需额外的 CUDA runtime 链接。

### `fcg-pass-fuse-full-flow`
一个 pattern-rewrite pass，将 `matmul → add → relu` 链（中间结果均单次使用）融合为单个 `fcg.fullflow`，在任何降级 pass 运行前减少中间张量的内存读写。

---

## 项目结构

```
cinder-mlir/
├── cinder-opt/
│   ├── include/cinder/Dialect/FCG/    # TableGen 定义与生成头文件
│   ├── lib/Dialect/FCG/
│   │   ├── IR/                        # 方言与算子定义、verifier
│   │   └── Transforms/                # 全部降级 pass
│   └── tools/cinder-opt/              # mlir-opt 风格的驱动工具
├── binding/
│   ├── c/include/                     # C ABI（CPU/GPU）
│   ├── fcg_export_cpu.mlir            # 测试内核（CPU）
│   ├── fcg_export_gpu.mlir            # 测试内核（GPU）+ CUDA runtime 辅助
│   └── python/                        # ctypes FFI 绑定
├── test/
│   ├── c/                             # C 基准测试驱动
│   └── python/                        # Python 基准测试驱动
├── scripts/llvm-mlir-install.mk       # 引导安装 LLVM/MLIR 工具链
├── CMakeLists.txt                     # 顶层构建
└── Makefile                           # 便捷命令
```

---

## 构建

* llvm-project 代码需要下载到代码目录
* cuda-11.8
* gcc-10/g++-10

Cinder 依赖固定版本的 LLVM/MLIR（见 `scripts/llvm-mlir-install.mk`）。先构建一次工具链，再以 `-DMLIR_TOOL_DIR` 指向安装好的 MLIR 配置本项目。

```bash
make llvm_mlir_install      # 构建并安装 LLVM/MLIR 到 ./llvm-mlir-bin
make                        # 配置、构建并安装 cinder 以及测试程序
```

构建产物包括 `test/bin/cinder-opt` 以及 `test/lib/` 下的基准库 `libFCGPass{Baseline,Vectorize,Linalg,GPU,NVVM}.so`。

---

## 运行基准测试

每个测试驱动都会将 MLIR 生成的内核与手写的 CPU 参考实现做正确性对比，并输出 GFLOPS 与 CPU/MLIR 加速比。

```bash
make test
```

或直接运行单个二进制：

```bash
test/bin/fcg-pass-vectorize 1024 1024 1024 2   # M N K repeat
python3 test/python/mainCPU.py --pass=Vectorize
```

GPU 驱动需要 CUDA 设备与 CUDA runtime/toolkit（默认 `sm_86`，可通过 `CINDER_CUDA_SMLEVEL` 覆盖）。

> 具体数值与硬件强相关，交由基准测试复现，README 有意不固化具体数字。`test.log` 为笔者笔记本上随机一次测试数据。
