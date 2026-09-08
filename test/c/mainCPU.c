#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <libgen.h>
#include "fcg_export_cpu.h"

#define EPSILON_F32 1e-5f
#define EPSILON_F16 1

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_random_f32(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static void fill_random_f16(_Float16* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = (_Float16)((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
}

static int check_matrices_f32(const float* C_cpu, const float* C_mlir, int M, int N, float eps) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = fabsf(C_cpu[i * N + j] - C_mlir[i * N + j]);
            if (diff > eps) {
                printf("Mismatch at [%d][%d]: CPU=%f, MLIR=%f, diff=%e, eps=%e\n",
                       i, j, C_cpu[i * N + j], C_mlir[i * N + j], diff, eps);
                return 0;
            }
        }
    }
    return 1;
}

static int check_matrices_f16(const _Float16* C_cpu, const _Float16* C_mlir, int M, int N, float eps) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = fabsf((float)C_cpu[i * N + j] - (float)C_mlir[i * N + j]);
            if (diff > eps) {
                printf("Mismatch at [%d][%d]: CPU=%f, MLIR=%f, diff=%e, eps=%e\n",
                       i, j, (float)C_cpu[i * N + j], (float)C_mlir[i * N + j], diff, eps);
                return 0;
            }
        }
    }
    return 1;
}

static void matmul_cpu_f32(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

static void add_cpu_f32(const float* A, const float* B, float* C, int M, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * N + j] = A[i * N + j] + B[i * N + j];
}

static void relu_cpu_f32(const float* X, float* Y, int M, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float x = X[i * N + j];
            Y[i * N + j] = x > 0.0f ? x : 0.0f;
        }
}

static void full_flow_cpu_f32(const float* A, const float* B, const float* bias,
                              float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            float val = sum + bias[i * N + j];
            C[i * N + j] = val > 0.0f ? val : 0.0f;
        }
    }
}

static void matmul_cpu_f16(const _Float16* A, const _Float16* B, _Float16* C,
                           int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += (float)A[i * K + k] * (float)B[k * N + j];
            C[i * N + j] = (_Float16)sum;
        }
}

static void add_cpu_f16(const _Float16* A, const _Float16* B, _Float16* C,
                        int M, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i * N + j] = (_Float16)((float)A[i * N + j] + (float)B[i * N + j]);
}

static void relu_cpu_f16(const _Float16* X, _Float16* Y, int M, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float x = (float)X[i * N + j];
            Y[i * N + j] = (_Float16)(x > 0.0f ? x : 0.0f);
        }
}

static void full_flow_cpu_f16(const _Float16* A, const _Float16* B,
                              const _Float16* bias, _Float16* C,
                              int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += (float)A[i * K + k] * (float)B[k * N + j];
            float val = sum + (float)bias[i * N + j];
            C[i * N + j] = (_Float16)(val > 0.0f ? val : 0.0f);
        }
    }
}

static double bench_cpu_matmul_f32(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f32(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f32(K, N);
    struct MemRef C_mem = fcg_memref_alloc_f32(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* A = (float*)A_mem.aligned;
    float* B = (float*)B_mem.aligned;
    float* C = (float*)C_mem.aligned;
    memset(C, 0, M * N * sizeof(float));
    fill_random_f32(A, M * K);
    fill_random_f32(B, K * N);

    for (int i = 0; i < 2; ++i) matmul_cpu_f32(A, B, C, M, N, K);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) matmul_cpu_f32(A, B, C, M, N, K);
    double elapsed = get_time_sec() - start;

    double flops = 2.0 * M * N * K * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f32(A_mem);
    fcg_memref_free_f32(B_mem);
    fcg_memref_free_f32(C_mem);
    return gflops;
}

static double bench_mlir_matmul_f32(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f32(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f32(K, N);
    struct MemRef C_mem = fcg_memref_alloc_f32(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* A = (float*)A_mem.aligned;
    float* B = (float*)B_mem.aligned;
    float* C_mlir = (float*)C_mem.aligned;
    memset(C_mlir, 0, M * N * sizeof(float));
    fill_random_f32(A, M * K);
    fill_random_f32(B, K * N);

    struct MemRef C_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!C_cpu_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* C_cpu = (float*)C_cpu_mem.aligned;
    memset(C_cpu, 0, M * N * sizeof(float));

    matmul_cpu_f32(A, B, C_cpu, M, N, K);
    fcg_matmul_f32(A_mem, B_mem, C_mem);
    if (!check_matrices_f32(C_cpu, C_mlir, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for matmul_f32\n");
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(C_mem);
        fcg_memref_free_f32(C_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_matmul_f32(A_mem, B_mem, C_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_matmul_f32(A_mem, B_mem, C_mem);
    double elapsed = get_time_sec() - start;

    double flops = 2.0 * M * N * K * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f32(A_mem);
    fcg_memref_free_f32(B_mem);
    fcg_memref_free_f32(C_mem);
    fcg_memref_free_f32(C_cpu_mem);
    return gflops;
}

static double bench_cpu_add_f32(int M, int N, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef B_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f32(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* A = (float*)A_mem.aligned;
    float* B = (float*)B_mem.aligned;
    float* C = (float*)C_mem.aligned;
    memset(C, 0, M * N * sizeof(float));
    fill_random_f32(A, M * N);
    fill_random_f32(B, M * N);

    for (int i = 0; i < 2; ++i) add_cpu_f32(A, B, C, M, N);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) add_cpu_f32(A, B, C, M, N);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f32(A_mem);
    fcg_memref_free_f32(B_mem);
    fcg_memref_free_f32(C_mem);
    return gflops;
}

static double bench_mlir_add_f32(int M, int N, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef B_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f32(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* A = (float*)A_mem.aligned;
    float* B = (float*)B_mem.aligned;
    float* C_mlir = (float*)C_mem.aligned;
    memset(C_mlir, 0, M * N * sizeof(float));
    fill_random_f32(A, M * N);
    fill_random_f32(B, M * N);

    struct MemRef C_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!C_cpu_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* C_cpu = (float*)C_cpu_mem.aligned;
    memset(C_cpu, 0, M * N * sizeof(float));

    add_cpu_f32(A, B, C_cpu, M, N);
    fcg_add_f32(A_mem, B_mem, C_mem);
    if (!check_matrices_f32(C_cpu, C_mlir, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for add_f32\n");
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(C_mem);
        fcg_memref_free_f32(C_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_add_f32(A_mem, B_mem, C_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_add_f32(A_mem, B_mem, C_mem);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f32(A_mem);
    fcg_memref_free_f32(B_mem);
    fcg_memref_free_f32(C_mem);
    fcg_memref_free_f32(C_cpu_mem);
    return gflops;
}

static double bench_cpu_relu_f32(int M, int N, int repeat) {
    struct MemRef X_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef Y_mem = fcg_memref_alloc_f32(M, N);
    if (!X_mem.aligned || !Y_mem.aligned) {
        fcg_memref_free_f32(X_mem);
        fcg_memref_free_f32(Y_mem);
        return 0.0;
    }
    float* X = (float*)X_mem.aligned;
    float* Y = (float*)Y_mem.aligned;
    memset(Y, 0, M * N * sizeof(float));
    fill_random_f32(X, M * N);

    for (int i = 0; i < 2; ++i) relu_cpu_f32(X, Y, M, N);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) relu_cpu_f32(X, Y, M, N);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f32(X_mem);
    fcg_memref_free_f32(Y_mem);
    return gflops;
}

static double bench_mlir_relu_f32(int M, int N, int repeat) {
    struct MemRef X_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef Y_mem = fcg_memref_alloc_f32(M, N);
    if (!X_mem.aligned || !Y_mem.aligned) {
        fcg_memref_free_f32(X_mem);
        fcg_memref_free_f32(Y_mem);
        return 0.0;
    }
    float* X = (float*)X_mem.aligned;
    float* Y_mlir = (float*)Y_mem.aligned;
    memset(Y_mlir, 0, M * N * sizeof(float));
    fill_random_f32(X, M * N);

    struct MemRef Y_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!Y_cpu_mem.aligned) {
        fcg_memref_free_f32(X_mem);
        fcg_memref_free_f32(Y_mem);
        return 0.0;
    }
    float* Y_cpu = (float*)Y_cpu_mem.aligned;
    memset(Y_cpu, 0, M * N * sizeof(float));

    relu_cpu_f32(X, Y_cpu, M, N);
    fcg_relu_f32(X_mem, Y_mem);
    if (!check_matrices_f32(Y_cpu, Y_mlir, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for relu_f32\n");
        fcg_memref_free_f32(X_mem);
        fcg_memref_free_f32(Y_mem);
        fcg_memref_free_f32(Y_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_relu_f32(X_mem, Y_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_relu_f32(X_mem, Y_mem);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f32(X_mem);
    fcg_memref_free_f32(Y_mem);
    fcg_memref_free_f32(Y_cpu_mem);
    return gflops;
}

static double bench_cpu_full_flow_f32(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f32(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f32(K, N);
    struct MemRef bias_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f32(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !bias_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(bias_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* A = (float*)A_mem.aligned;
    float* B = (float*)B_mem.aligned;
    float* bias = (float*)bias_mem.aligned;
    float* C = (float*)C_mem.aligned;
    memset(C, 0, M * N * sizeof(float));
    fill_random_f32(A, M * K);
    fill_random_f32(B, K * N);
    fill_random_f32(bias, M * N);

    for (int i = 0; i < 2; ++i) full_flow_cpu_f32(A, B, bias, C, M, N, K);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) full_flow_cpu_f32(A, B, bias, C, M, N, K);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f32(A_mem);
    fcg_memref_free_f32(B_mem);
    fcg_memref_free_f32(bias_mem);
    fcg_memref_free_f32(C_mem);
    return gflops;
}

static double bench_mlir_full_flow_f32(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f32(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f32(K, N);
    struct MemRef bias_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f32(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !bias_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(bias_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* A = (float*)A_mem.aligned;
    float* B = (float*)B_mem.aligned;
    float* bias = (float*)bias_mem.aligned;
    float* C_mlir = (float*)C_mem.aligned;
    memset(C_mlir, 0, M * N * sizeof(float));
    fill_random_f32(A, M * K);
    fill_random_f32(B, K * N);
    fill_random_f32(bias, M * N);

    struct MemRef C_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!C_cpu_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(bias_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* C_cpu = (float*)C_cpu_mem.aligned;
    memset(C_cpu, 0, M * N * sizeof(float));

    full_flow_cpu_f32(A, B, bias, C_cpu, M, N, K);
    fcg_full_flow_f32(A_mem, B_mem, bias_mem, C_mem);
    if (!check_matrices_f32(C_cpu, C_mlir, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for full_flow_f32\n");
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(bias_mem);
        fcg_memref_free_f32(C_mem);
        fcg_memref_free_f32(C_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_full_flow_f32(A_mem, B_mem, bias_mem, C_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_full_flow_f32(A_mem, B_mem, bias_mem, C_mem);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f32(A_mem);
    fcg_memref_free_f32(B_mem);
    fcg_memref_free_f32(bias_mem);
    fcg_memref_free_f32(C_mem);
    fcg_memref_free_f32(C_cpu_mem);
    return gflops;
}

static double bench_mlir_fusion_full_flow_f32(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f32(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f32(K, N);
    struct MemRef bias_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f32(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !bias_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(bias_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* A = (float*)A_mem.aligned;
    float* B = (float*)B_mem.aligned;
    float* bias = (float*)bias_mem.aligned;
    float* C_mlir = (float*)C_mem.aligned;
    memset(C_mlir, 0, M * N * sizeof(float));
    fill_random_f32(A, M * K);
    fill_random_f32(B, K * N);
    fill_random_f32(bias, M * N);

    struct MemRef C_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!C_cpu_mem.aligned) {
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(bias_mem);
        fcg_memref_free_f32(C_mem);
        return 0.0;
    }
    float* C_cpu = (float*)C_cpu_mem.aligned;
    memset(C_cpu, 0, M * N * sizeof(float));

    full_flow_cpu_f32(A, B, bias, C_cpu, M, N, K);
    fcg_fusion_full_flow_f32(A_mem, B_mem, bias_mem, C_mem);
    if (!check_matrices_f32(C_cpu, C_mlir, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for fusion_full_flow_f32\n");
        fcg_memref_free_f32(A_mem);
        fcg_memref_free_f32(B_mem);
        fcg_memref_free_f32(bias_mem);
        fcg_memref_free_f32(C_mem);
        fcg_memref_free_f32(C_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_fusion_full_flow_f32(A_mem, B_mem, bias_mem, C_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_fusion_full_flow_f32(A_mem, B_mem, bias_mem, C_mem);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f32(A_mem);
    fcg_memref_free_f32(B_mem);
    fcg_memref_free_f32(bias_mem);
    fcg_memref_free_f32(C_mem);
    fcg_memref_free_f32(C_cpu_mem);
    return gflops;
}

static double bench_cpu_matmul_f16(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f16(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f16(K, N);
    struct MemRef C_mem = fcg_memref_alloc_f16(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* A = (_Float16*)A_mem.aligned;
    _Float16* B = (_Float16*)B_mem.aligned;
    _Float16* C = (_Float16*)C_mem.aligned;
    memset(C, 0, M * N * sizeof(_Float16));
    fill_random_f16(A, M * K);
    fill_random_f16(B, K * N);

    for (int i = 0; i < 2; ++i) matmul_cpu_f16(A, B, C, M, N, K);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) matmul_cpu_f16(A, B, C, M, N, K);
    double elapsed = get_time_sec() - start;

    double flops = 2.0 * M * N * K * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f16(A_mem);
    fcg_memref_free_f16(B_mem);
    fcg_memref_free_f16(C_mem);
    return gflops;
}

static double bench_mlir_matmul_f16(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f16(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f16(K, N);
    struct MemRef C_mem = fcg_memref_alloc_f16(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* A = (_Float16*)A_mem.aligned;
    _Float16* B = (_Float16*)B_mem.aligned;
    _Float16* C_mlir = (_Float16*)C_mem.aligned;
    memset(C_mlir, 0, M * N * sizeof(_Float16));
    fill_random_f16(A, M * K);
    fill_random_f16(B, K * N);

    struct MemRef C_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!C_cpu_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* C_cpu = (_Float16*)C_cpu_mem.aligned;
    memset(C_cpu, 0, M * N * sizeof(_Float16));

    matmul_cpu_f16(A, B, C_cpu, M, N, K);
    fcg_matmul_f16(A_mem, B_mem, C_mem);
    if (!check_matrices_f16(C_cpu, C_mlir, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for matmul_f16\n");
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(C_mem);
        fcg_memref_free_f16(C_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_matmul_f16(A_mem, B_mem, C_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_matmul_f16(A_mem, B_mem, C_mem);
    double elapsed = get_time_sec() - start;

    double flops = 2.0 * M * N * K * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f16(A_mem);
    fcg_memref_free_f16(B_mem);
    fcg_memref_free_f16(C_mem);
    fcg_memref_free_f16(C_cpu_mem);
    return gflops;
}

static double bench_cpu_add_f16(int M, int N, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef B_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f16(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* A = (_Float16*)A_mem.aligned;
    _Float16* B = (_Float16*)B_mem.aligned;
    _Float16* C = (_Float16*)C_mem.aligned;
    memset(C, 0, M * N * sizeof(_Float16));
    fill_random_f16(A, M * N);
    fill_random_f16(B, M * N);

    for (int i = 0; i < 2; ++i) add_cpu_f16(A, B, C, M, N);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) add_cpu_f16(A, B, C, M, N);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f16(A_mem);
    fcg_memref_free_f16(B_mem);
    fcg_memref_free_f16(C_mem);
    return gflops;
}

static double bench_mlir_add_f16(int M, int N, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef B_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f16(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* A = (_Float16*)A_mem.aligned;
    _Float16* B = (_Float16*)B_mem.aligned;
    _Float16* C_mlir = (_Float16*)C_mem.aligned;
    memset(C_mlir, 0, M * N * sizeof(_Float16));
    fill_random_f16(A, M * N);
    fill_random_f16(B, M * N);

    struct MemRef C_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!C_cpu_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* C_cpu = (_Float16*)C_cpu_mem.aligned;
    memset(C_cpu, 0, M * N * sizeof(_Float16));

    add_cpu_f16(A, B, C_cpu, M, N);
    fcg_add_f16(A_mem, B_mem, C_mem);
    if (!check_matrices_f16(C_cpu, C_mlir, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for add_f16\n");
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(C_mem);
        fcg_memref_free_f16(C_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_add_f16(A_mem, B_mem, C_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_add_f16(A_mem, B_mem, C_mem);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f16(A_mem);
    fcg_memref_free_f16(B_mem);
    fcg_memref_free_f16(C_mem);
    fcg_memref_free_f16(C_cpu_mem);
    return gflops;
}

static double bench_cpu_relu_f16(int M, int N, int repeat) {
    struct MemRef X_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef Y_mem = fcg_memref_alloc_f16(M, N);
    if (!X_mem.aligned || !Y_mem.aligned) {
        fcg_memref_free_f16(X_mem);
        fcg_memref_free_f16(Y_mem);
        return 0.0;
    }
    _Float16* X = (_Float16*)X_mem.aligned;
    _Float16* Y = (_Float16*)Y_mem.aligned;
    memset(Y, 0, M * N * sizeof(_Float16));
    fill_random_f16(X, M * N);

    for (int i = 0; i < 2; ++i) relu_cpu_f16(X, Y, M, N);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) relu_cpu_f16(X, Y, M, N);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f16(X_mem);
    fcg_memref_free_f16(Y_mem);
    return gflops;
}

static double bench_mlir_relu_f16(int M, int N, int repeat) {
    struct MemRef X_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef Y_mem = fcg_memref_alloc_f16(M, N);
    if (!X_mem.aligned || !Y_mem.aligned) {
        fcg_memref_free_f16(X_mem);
        fcg_memref_free_f16(Y_mem);
        return 0.0;
    }
    _Float16* X = (_Float16*)X_mem.aligned;
    _Float16* Y_mlir = (_Float16*)Y_mem.aligned;
    memset(Y_mlir, 0, M * N * sizeof(_Float16));
    fill_random_f16(X, M * N);

    struct MemRef Y_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!Y_cpu_mem.aligned) {
        fcg_memref_free_f16(X_mem);
        fcg_memref_free_f16(Y_mem);
        return 0.0;
    }
    _Float16* Y_cpu = (_Float16*)Y_cpu_mem.aligned;
    memset(Y_cpu, 0, M * N * sizeof(_Float16));

    relu_cpu_f16(X, Y_cpu, M, N);
    fcg_relu_f16(X_mem, Y_mem);
    if (!check_matrices_f16(Y_cpu, Y_mlir, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for relu_f16\n");
        fcg_memref_free_f16(X_mem);
        fcg_memref_free_f16(Y_mem);
        fcg_memref_free_f16(Y_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_relu_f16(X_mem, Y_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_relu_f16(X_mem, Y_mem);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f16(X_mem);
    fcg_memref_free_f16(Y_mem);
    fcg_memref_free_f16(Y_cpu_mem);
    return gflops;
}

static double bench_cpu_full_flow_f16(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f16(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f16(K, N);
    struct MemRef bias_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f16(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !bias_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(bias_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* A = (_Float16*)A_mem.aligned;
    _Float16* B = (_Float16*)B_mem.aligned;
    _Float16* bias = (_Float16*)bias_mem.aligned;
    _Float16* C = (_Float16*)C_mem.aligned;
    memset(C, 0, M * N * sizeof(_Float16));
    fill_random_f16(A, M * K);
    fill_random_f16(B, K * N);
    fill_random_f16(bias, M * N);

    for (int i = 0; i < 2; ++i) full_flow_cpu_f16(A, B, bias, C, M, N, K);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) full_flow_cpu_f16(A, B, bias, C, M, N, K);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f16(A_mem);
    fcg_memref_free_f16(B_mem);
    fcg_memref_free_f16(bias_mem);
    fcg_memref_free_f16(C_mem);
    return gflops;
}

static double bench_mlir_full_flow_f16(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f16(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f16(K, N);
    struct MemRef bias_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f16(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !bias_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(bias_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* A = (_Float16*)A_mem.aligned;
    _Float16* B = (_Float16*)B_mem.aligned;
    _Float16* bias = (_Float16*)bias_mem.aligned;
    _Float16* C_mlir = (_Float16*)C_mem.aligned;
    memset(C_mlir, 0, M * N * sizeof(_Float16));
    fill_random_f16(A, M * K);
    fill_random_f16(B, K * N);
    fill_random_f16(bias, M * N);

    struct MemRef C_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!C_cpu_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(bias_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* C_cpu = (_Float16*)C_cpu_mem.aligned;
    memset(C_cpu, 0, M * N * sizeof(_Float16));

    full_flow_cpu_f16(A, B, bias, C_cpu, M, N, K);
    fcg_full_flow_f16(A_mem, B_mem, bias_mem, C_mem);
    if (!check_matrices_f16(C_cpu, C_mlir, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for full_flow_f16\n");
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(bias_mem);
        fcg_memref_free_f16(C_mem);
        fcg_memref_free_f16(C_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_full_flow_f16(A_mem, B_mem, bias_mem, C_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_full_flow_f16(A_mem, B_mem, bias_mem, C_mem);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f16(A_mem);
    fcg_memref_free_f16(B_mem);
    fcg_memref_free_f16(bias_mem);
    fcg_memref_free_f16(C_mem);
    fcg_memref_free_f16(C_cpu_mem);
    return gflops;
}

static double bench_mlir_fusion_full_flow_f16(int M, int N, int K, int repeat) {
    struct MemRef A_mem = fcg_memref_alloc_f16(M, K);
    struct MemRef B_mem = fcg_memref_alloc_f16(K, N);
    struct MemRef bias_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef C_mem = fcg_memref_alloc_f16(M, N);
    if (!A_mem.aligned || !B_mem.aligned || !bias_mem.aligned || !C_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(bias_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* A = (_Float16*)A_mem.aligned;
    _Float16* B = (_Float16*)B_mem.aligned;
    _Float16* bias = (_Float16*)bias_mem.aligned;
    _Float16* C_mlir = (_Float16*)C_mem.aligned;
    memset(C_mlir, 0, M * N * sizeof(_Float16));
    fill_random_f16(A, M * K);
    fill_random_f16(B, K * N);
    fill_random_f16(bias, M * N);

    struct MemRef C_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!C_cpu_mem.aligned) {
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(bias_mem);
        fcg_memref_free_f16(C_mem);
        return 0.0;
    }
    _Float16* C_cpu = (_Float16*)C_cpu_mem.aligned;
    memset(C_cpu, 0, M * N * sizeof(_Float16));

    full_flow_cpu_f16(A, B, bias, C_cpu, M, N, K);
    fcg_fusion_full_flow_f16(A_mem, B_mem, bias_mem, C_mem);
    if (!check_matrices_f16(C_cpu, C_mlir, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for fusion_full_flow_f16\n");
        fcg_memref_free_f16(A_mem);
        fcg_memref_free_f16(B_mem);
        fcg_memref_free_f16(bias_mem);
        fcg_memref_free_f16(C_mem);
        fcg_memref_free_f16(C_cpu_mem);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_fusion_full_flow_f16(A_mem, B_mem, bias_mem, C_mem);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_fusion_full_flow_f16(A_mem, B_mem, bias_mem, C_mem);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);
    fcg_memref_free_f16(A_mem);
    fcg_memref_free_f16(B_mem);
    fcg_memref_free_f16(bias_mem);
    fcg_memref_free_f16(C_mem);
    fcg_memref_free_f16(C_cpu_mem);
    return gflops;
}

int main(int argc, char **argv) {
    int M = 128, N = 128, K = 128;
    int repeat = 5;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) repeat = atoi(argv[4]);

    char *exe = basename(argv[0]);
    char *pass_name   = "unknown";
    char *pass_prefix = "fcg-pass-";
    if (strncmp(exe, pass_prefix, strlen(pass_prefix)) == 0) {
        pass_name = exe + strlen(pass_prefix);
    }

    printf("=== Performance Comparison (CPU vs MLIR/C) ===\n");
    printf("MLIR Pass: %s\n", pass_name);
    printf("M=%d, N=%d, K=%d, repeat=%d\n\n", M, N, K, repeat);

    printf("---- f32 ----\n");
    double cpu, mlir;
    cpu = bench_cpu_matmul_f32(M, N, K, repeat);
    mlir = bench_mlir_matmul_f32(M, N, K, repeat);
    printf("matmul      : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    cpu = bench_cpu_add_f32(M, N, repeat);
    mlir = bench_mlir_add_f32(M, N, repeat);
    printf("add         : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    cpu = bench_cpu_relu_f32(M, N, repeat);
    mlir = bench_mlir_relu_f32(M, N, repeat);
    printf("relu        : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    cpu = bench_cpu_full_flow_f32(M, N, K, repeat);
    mlir = bench_mlir_full_flow_f32(M, N, K, repeat);
    printf("full_flow   : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    mlir = bench_mlir_fusion_full_flow_f32(M, N, K, repeat);
    printf("fusion_flow : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    printf("\n---- f16 ----\n");
    cpu = bench_cpu_matmul_f16(M, N, K, repeat);
    mlir = bench_mlir_matmul_f16(M, N, K, repeat);
    printf("matmul      : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    cpu = bench_cpu_add_f16(M, N, repeat);
    mlir = bench_mlir_add_f16(M, N, repeat);
    printf("add         : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    cpu = bench_cpu_relu_f16(M, N, repeat);
    mlir = bench_mlir_relu_f16(M, N, repeat);
    printf("relu        : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    cpu = bench_cpu_full_flow_f16(M, N, K, repeat);
    mlir = bench_mlir_full_flow_f16(M, N, K, repeat);
    printf("full_flow   : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    mlir = bench_mlir_fusion_full_flow_f16(M, N, K, repeat);
    printf("fusion_flow : CPU = %8.2f GFLOPS, MLIR = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, mlir, cpu / mlir);

    return 0;
}
