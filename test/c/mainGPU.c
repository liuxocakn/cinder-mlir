#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <libgen.h>
#include "fcg_export_gpu.h"

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

static int check_matrices_f32(const float* C_cpu, const float* C_gpu, int M, int N, float eps) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = fabsf(C_cpu[i * N + j] - C_gpu[i * N + j]);
            if (diff > eps) {
                printf("Mismatch at [%d][%d]: CPU=%f, GPU=%f, diff=%e, eps=%e\n",
                       i, j, C_cpu[i * N + j], C_gpu[i * N + j], diff, eps);
                return 0;
            }
        }
    }
    return 1;
}

static int check_matrices_f16(const _Float16* C_cpu, const _Float16* C_gpu, int M, int N, float eps) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = fabsf((float)C_cpu[i * N + j] - (float)C_gpu[i * N + j]);
            if (diff > eps) {
                printf("Mismatch at [%d][%d]: CPU=%f, GPU=%f, diff=%e, eps=%e\n",
                       i, j, (float)C_cpu[i * N + j], (float)C_gpu[i * N + j], diff, eps);
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

static double bench_gpu_matmul_f32(int M, int N, int K, int repeat) {
    struct MemRef h_A_mem = fcg_memref_alloc_f32(M, K);
    struct MemRef h_B_mem = fcg_memref_alloc_f32(K, N);
    struct MemRef h_C_gpu_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_C_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!h_A_mem.aligned || !h_B_mem.aligned || !h_C_gpu_mem.aligned || !h_C_cpu_mem.aligned) {
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        return 0.0;
    }
    float* h_A = (float*)h_A_mem.aligned;
    float* h_B = (float*)h_B_mem.aligned;
    float* h_C_gpu = (float*)h_C_gpu_mem.aligned;
    float* h_C_cpu = (float*)h_C_cpu_mem.aligned;

    fill_random_f32(h_A, M * K);
    fill_random_f32(h_B, K * N);
    memset(h_C_gpu, 0, M * N * sizeof(float));
    memset(h_C_cpu, 0, M * N * sizeof(float));

    struct MemRef d_A = fcg_gpu_memref_alloc_f32(M, K);
    struct MemRef d_B = fcg_gpu_memref_alloc_f32(K, N);
    struct MemRef d_C = fcg_gpu_memref_alloc_f32(M, N);
    if (!d_A.aligned || !d_B.aligned || !d_C.aligned) {
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        fcg_gpu_memref_free_f32(d_A);
        fcg_gpu_memref_free_f32(d_B);
        fcg_gpu_memref_free_f32(d_C);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f32(h_A_mem, d_A);
    fcg_gpu_memref_copy_h2d_f32(h_B_mem, d_B);

    matmul_cpu_f32(h_A, h_B, h_C_cpu, M, N, K);
    fcg_matmul_f32(d_A, d_B, d_C);
    fcg_gpu_memref_copy_d2h_f32(d_C, h_C_gpu_mem);

    if (!check_matrices_f32(h_C_cpu, h_C_gpu, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for matmul_f32 (GPU)\n");
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        fcg_gpu_memref_free_f32(d_A);
        fcg_gpu_memref_free_f32(d_B);
        fcg_gpu_memref_free_f32(d_C);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_matmul_f32(d_A, d_B, d_C);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_matmul_f32(d_A, d_B, d_C);
    double elapsed = get_time_sec() - start;

    double flops = 2.0 * M * N * K * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f32(h_A_mem);
    fcg_memref_free_f32(h_B_mem);
    fcg_memref_free_f32(h_C_gpu_mem);
    fcg_memref_free_f32(h_C_cpu_mem);
    fcg_gpu_memref_free_f32(d_A);
    fcg_gpu_memref_free_f32(d_B);
    fcg_gpu_memref_free_f32(d_C);
    return gflops;
}

static double bench_gpu_add_f32(int M, int N, int repeat) {
    struct MemRef h_A_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_B_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_C_gpu_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_C_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!h_A_mem.aligned || !h_B_mem.aligned || !h_C_gpu_mem.aligned || !h_C_cpu_mem.aligned) {
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        return 0.0;
    }
    float* h_A = (float*)h_A_mem.aligned;
    float* h_B = (float*)h_B_mem.aligned;
    float* h_C_gpu = (float*)h_C_gpu_mem.aligned;
    float* h_C_cpu = (float*)h_C_cpu_mem.aligned;

    fill_random_f32(h_A, M * N);
    fill_random_f32(h_B, M * N);
    memset(h_C_gpu, 0, M * N * sizeof(float));
    memset(h_C_cpu, 0, M * N * sizeof(float));

    struct MemRef d_A = fcg_gpu_memref_alloc_f32(M, N);
    struct MemRef d_B = fcg_gpu_memref_alloc_f32(M, N);
    struct MemRef d_C = fcg_gpu_memref_alloc_f32(M, N);
    if (!d_A.aligned || !d_B.aligned || !d_C.aligned) {
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        fcg_gpu_memref_free_f32(d_A);
        fcg_gpu_memref_free_f32(d_B);
        fcg_gpu_memref_free_f32(d_C);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f32(h_A_mem, d_A);
    fcg_gpu_memref_copy_h2d_f32(h_B_mem, d_B);

    add_cpu_f32(h_A, h_B, h_C_cpu, M, N);
    fcg_add_f32(d_A, d_B, d_C);
    fcg_gpu_memref_copy_d2h_f32(d_C, h_C_gpu_mem);

    if (!check_matrices_f32(h_C_cpu, h_C_gpu, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for add_f32 (GPU)\n");
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        fcg_gpu_memref_free_f32(d_A);
        fcg_gpu_memref_free_f32(d_B);
        fcg_gpu_memref_free_f32(d_C);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_add_f32(d_A, d_B, d_C);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_add_f32(d_A, d_B, d_C);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f32(h_A_mem);
    fcg_memref_free_f32(h_B_mem);
    fcg_memref_free_f32(h_C_gpu_mem);
    fcg_memref_free_f32(h_C_cpu_mem);
    fcg_gpu_memref_free_f32(d_A);
    fcg_gpu_memref_free_f32(d_B);
    fcg_gpu_memref_free_f32(d_C);
    return gflops;
}

static double bench_gpu_relu_f32(int M, int N, int repeat) {
    struct MemRef h_X_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_Y_gpu_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_Y_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!h_X_mem.aligned || !h_Y_gpu_mem.aligned || !h_Y_cpu_mem.aligned) {
        fcg_memref_free_f32(h_X_mem);
        fcg_memref_free_f32(h_Y_gpu_mem);
        fcg_memref_free_f32(h_Y_cpu_mem);
        return 0.0;
    }
    float* h_X = (float*)h_X_mem.aligned;
    float* h_Y_gpu = (float*)h_Y_gpu_mem.aligned;
    float* h_Y_cpu = (float*)h_Y_cpu_mem.aligned;

    fill_random_f32(h_X, M * N);
    memset(h_Y_gpu, 0, M * N * sizeof(float));
    memset(h_Y_cpu, 0, M * N * sizeof(float));

    struct MemRef d_X = fcg_gpu_memref_alloc_f32(M, N);
    struct MemRef d_Y = fcg_gpu_memref_alloc_f32(M, N);
    if (!d_X.aligned || !d_Y.aligned) {
        fcg_memref_free_f32(h_X_mem);
        fcg_memref_free_f32(h_Y_gpu_mem);
        fcg_memref_free_f32(h_Y_cpu_mem);
        fcg_gpu_memref_free_f32(d_X);
        fcg_gpu_memref_free_f32(d_Y);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f32(h_X_mem, d_X);

    relu_cpu_f32(h_X, h_Y_cpu, M, N);
    fcg_relu_f32(d_X, d_Y);
    fcg_gpu_memref_copy_d2h_f32(d_Y, h_Y_gpu_mem);

    if (!check_matrices_f32(h_Y_cpu, h_Y_gpu, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for relu_f32 (GPU)\n");
        fcg_memref_free_f32(h_X_mem);
        fcg_memref_free_f32(h_Y_gpu_mem);
        fcg_memref_free_f32(h_Y_cpu_mem);
        fcg_gpu_memref_free_f32(d_X);
        fcg_gpu_memref_free_f32(d_Y);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_relu_f32(d_X, d_Y);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_relu_f32(d_X, d_Y);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f32(h_X_mem);
    fcg_memref_free_f32(h_Y_gpu_mem);
    fcg_memref_free_f32(h_Y_cpu_mem);
    fcg_gpu_memref_free_f32(d_X);
    fcg_gpu_memref_free_f32(d_Y);
    return gflops;
}

static double bench_gpu_full_flow_f32(int M, int N, int K, int repeat) {
    struct MemRef h_A_mem = fcg_memref_alloc_f32(M, K);
    struct MemRef h_B_mem = fcg_memref_alloc_f32(K, N);
    struct MemRef h_bias_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_C_gpu_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_C_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!h_A_mem.aligned || !h_B_mem.aligned || !h_bias_mem.aligned ||
        !h_C_gpu_mem.aligned || !h_C_cpu_mem.aligned) {
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_bias_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        return 0.0;
    }
    float* h_A = (float*)h_A_mem.aligned;
    float* h_B = (float*)h_B_mem.aligned;
    float* h_bias = (float*)h_bias_mem.aligned;
    float* h_C_gpu = (float*)h_C_gpu_mem.aligned;
    float* h_C_cpu = (float*)h_C_cpu_mem.aligned;

    fill_random_f32(h_A, M * K);
    fill_random_f32(h_B, K * N);
    fill_random_f32(h_bias, M * N);
    memset(h_C_gpu, 0, M * N * sizeof(float));
    memset(h_C_cpu, 0, M * N * sizeof(float));

    struct MemRef d_A = fcg_gpu_memref_alloc_f32(M, K);
    struct MemRef d_B = fcg_gpu_memref_alloc_f32(K, N);
    struct MemRef d_bias = fcg_gpu_memref_alloc_f32(M, N);
    struct MemRef d_C = fcg_gpu_memref_alloc_f32(M, N);
    if (!d_A.aligned || !d_B.aligned || !d_bias.aligned || !d_C.aligned) {
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_bias_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        fcg_gpu_memref_free_f32(d_A);
        fcg_gpu_memref_free_f32(d_B);
        fcg_gpu_memref_free_f32(d_bias);
        fcg_gpu_memref_free_f32(d_C);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f32(h_A_mem, d_A);
    fcg_gpu_memref_copy_h2d_f32(h_B_mem, d_B);
    fcg_gpu_memref_copy_h2d_f32(h_bias_mem, d_bias);

    full_flow_cpu_f32(h_A, h_B, h_bias, h_C_cpu, M, N, K);
    fcg_full_flow_f32(d_A, d_B, d_bias, d_C);
    fcg_gpu_memref_copy_d2h_f32(d_C, h_C_gpu_mem);

    if (!check_matrices_f32(h_C_cpu, h_C_gpu, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for full_flow_f32 (GPU)\n");
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_bias_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        fcg_gpu_memref_free_f32(d_A);
        fcg_gpu_memref_free_f32(d_B);
        fcg_gpu_memref_free_f32(d_bias);
        fcg_gpu_memref_free_f32(d_C);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_full_flow_f32(d_A, d_B, d_bias, d_C);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_full_flow_f32(d_A, d_B, d_bias, d_C);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f32(h_A_mem);
    fcg_memref_free_f32(h_B_mem);
    fcg_memref_free_f32(h_bias_mem);
    fcg_memref_free_f32(h_C_gpu_mem);
    fcg_memref_free_f32(h_C_cpu_mem);
    fcg_gpu_memref_free_f32(d_A);
    fcg_gpu_memref_free_f32(d_B);
    fcg_gpu_memref_free_f32(d_bias);
    fcg_gpu_memref_free_f32(d_C);
    return gflops;
}

static double bench_gpu_fusion_full_flow_f32(int M, int N, int K, int repeat) {
    struct MemRef h_A_mem = fcg_memref_alloc_f32(M, K);
    struct MemRef h_B_mem = fcg_memref_alloc_f32(K, N);
    struct MemRef h_bias_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_C_gpu_mem = fcg_memref_alloc_f32(M, N);
    struct MemRef h_C_cpu_mem = fcg_memref_alloc_f32(M, N);
    if (!h_A_mem.aligned || !h_B_mem.aligned || !h_bias_mem.aligned ||
        !h_C_gpu_mem.aligned || !h_C_cpu_mem.aligned) {
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_bias_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        return 0.0;
    }
    float* h_A = (float*)h_A_mem.aligned;
    float* h_B = (float*)h_B_mem.aligned;
    float* h_bias = (float*)h_bias_mem.aligned;
    float* h_C_gpu = (float*)h_C_gpu_mem.aligned;
    float* h_C_cpu = (float*)h_C_cpu_mem.aligned;

    fill_random_f32(h_A, M * K);
    fill_random_f32(h_B, K * N);
    fill_random_f32(h_bias, M * N);
    memset(h_C_gpu, 0, M * N * sizeof(float));
    memset(h_C_cpu, 0, M * N * sizeof(float));

    struct MemRef d_A = fcg_gpu_memref_alloc_f32(M, K);
    struct MemRef d_B = fcg_gpu_memref_alloc_f32(K, N);
    struct MemRef d_bias = fcg_gpu_memref_alloc_f32(M, N);
    struct MemRef d_C = fcg_gpu_memref_alloc_f32(M, N);
    if (!d_A.aligned || !d_B.aligned || !d_bias.aligned || !d_C.aligned) {
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_bias_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        fcg_gpu_memref_free_f32(d_A);
        fcg_gpu_memref_free_f32(d_B);
        fcg_gpu_memref_free_f32(d_bias);
        fcg_gpu_memref_free_f32(d_C);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f32(h_A_mem, d_A);
    fcg_gpu_memref_copy_h2d_f32(h_B_mem, d_B);
    fcg_gpu_memref_copy_h2d_f32(h_bias_mem, d_bias);

    full_flow_cpu_f32(h_A, h_B, h_bias, h_C_cpu, M, N, K);
    fcg_fusion_full_flow_f32(d_A, d_B, d_bias, d_C);
    fcg_gpu_memref_copy_d2h_f32(d_C, h_C_gpu_mem);

    if (!check_matrices_f32(h_C_cpu, h_C_gpu, M, N, EPSILON_F32)) {
        fprintf(stderr, "Correctness check FAILED for fusion_full_flow_f32 (GPU)\n");
        fcg_memref_free_f32(h_A_mem);
        fcg_memref_free_f32(h_B_mem);
        fcg_memref_free_f32(h_bias_mem);
        fcg_memref_free_f32(h_C_gpu_mem);
        fcg_memref_free_f32(h_C_cpu_mem);
        fcg_gpu_memref_free_f32(d_A);
        fcg_gpu_memref_free_f32(d_B);
        fcg_gpu_memref_free_f32(d_bias);
        fcg_gpu_memref_free_f32(d_C);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_fusion_full_flow_f32(d_A, d_B, d_bias, d_C);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_fusion_full_flow_f32(d_A, d_B, d_bias, d_C);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f32(h_A_mem);
    fcg_memref_free_f32(h_B_mem);
    fcg_memref_free_f32(h_bias_mem);
    fcg_memref_free_f32(h_C_gpu_mem);
    fcg_memref_free_f32(h_C_cpu_mem);
    fcg_gpu_memref_free_f32(d_A);
    fcg_gpu_memref_free_f32(d_B);
    fcg_gpu_memref_free_f32(d_bias);
    fcg_gpu_memref_free_f32(d_C);
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

static double bench_gpu_matmul_f16(int M, int N, int K, int repeat) {
    struct MemRef h_A_mem = fcg_memref_alloc_f16(M, K);
    struct MemRef h_B_mem = fcg_memref_alloc_f16(K, N);
    struct MemRef h_C_gpu_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_C_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!h_A_mem.aligned || !h_B_mem.aligned || !h_C_gpu_mem.aligned || !h_C_cpu_mem.aligned) {
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        return 0.0;
    }
    _Float16* h_A = (_Float16*)h_A_mem.aligned;
    _Float16* h_B = (_Float16*)h_B_mem.aligned;
    _Float16* h_C_gpu = (_Float16*)h_C_gpu_mem.aligned;
    _Float16* h_C_cpu = (_Float16*)h_C_cpu_mem.aligned;

    fill_random_f16(h_A, M * K);
    fill_random_f16(h_B, K * N);
    memset(h_C_gpu, 0, M * N * sizeof(_Float16));
    memset(h_C_cpu, 0, M * N * sizeof(_Float16));

    struct MemRef d_A = fcg_gpu_memref_alloc_f16(M, K);
    struct MemRef d_B = fcg_gpu_memref_alloc_f16(K, N);
    struct MemRef d_C = fcg_gpu_memref_alloc_f16(M, N);
    if (!d_A.aligned || !d_B.aligned || !d_C.aligned) {
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        fcg_gpu_memref_free_f16(d_A);
        fcg_gpu_memref_free_f16(d_B);
        fcg_gpu_memref_free_f16(d_C);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f16(h_A_mem, d_A);
    fcg_gpu_memref_copy_h2d_f16(h_B_mem, d_B);

    matmul_cpu_f16(h_A, h_B, h_C_cpu, M, N, K);
    fcg_matmul_f16(d_A, d_B, d_C);
    fcg_gpu_memref_copy_d2h_f16(d_C, h_C_gpu_mem);

    if (!check_matrices_f16(h_C_cpu, h_C_gpu, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for matmul_f16 (GPU)\n");
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        fcg_gpu_memref_free_f16(d_A);
        fcg_gpu_memref_free_f16(d_B);
        fcg_gpu_memref_free_f16(d_C);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_matmul_f16(d_A, d_B, d_C);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_matmul_f16(d_A, d_B, d_C);
    double elapsed = get_time_sec() - start;

    double flops = 2.0 * M * N * K * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f16(h_A_mem);
    fcg_memref_free_f16(h_B_mem);
    fcg_memref_free_f16(h_C_gpu_mem);
    fcg_memref_free_f16(h_C_cpu_mem);
    fcg_gpu_memref_free_f16(d_A);
    fcg_gpu_memref_free_f16(d_B);
    fcg_gpu_memref_free_f16(d_C);
    return gflops;
}

static double bench_gpu_add_f16(int M, int N, int repeat) {
    struct MemRef h_A_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_B_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_C_gpu_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_C_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!h_A_mem.aligned || !h_B_mem.aligned || !h_C_gpu_mem.aligned || !h_C_cpu_mem.aligned) {
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        return 0.0;
    }
    _Float16* h_A = (_Float16*)h_A_mem.aligned;
    _Float16* h_B = (_Float16*)h_B_mem.aligned;
    _Float16* h_C_gpu = (_Float16*)h_C_gpu_mem.aligned;
    _Float16* h_C_cpu = (_Float16*)h_C_cpu_mem.aligned;

    fill_random_f16(h_A, M * N);
    fill_random_f16(h_B, M * N);
    memset(h_C_gpu, 0, M * N * sizeof(_Float16));
    memset(h_C_cpu, 0, M * N * sizeof(_Float16));

    struct MemRef d_A = fcg_gpu_memref_alloc_f16(M, N);
    struct MemRef d_B = fcg_gpu_memref_alloc_f16(M, N);
    struct MemRef d_C = fcg_gpu_memref_alloc_f16(M, N);
    if (!d_A.aligned || !d_B.aligned || !d_C.aligned) {
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        fcg_gpu_memref_free_f16(d_A);
        fcg_gpu_memref_free_f16(d_B);
        fcg_gpu_memref_free_f16(d_C);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f16(h_A_mem, d_A);
    fcg_gpu_memref_copy_h2d_f16(h_B_mem, d_B);

    add_cpu_f16(h_A, h_B, h_C_cpu, M, N);
    fcg_add_f16(d_A, d_B, d_C);
    fcg_gpu_memref_copy_d2h_f16(d_C, h_C_gpu_mem);

    if (!check_matrices_f16(h_C_cpu, h_C_gpu, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for add_f16 (GPU)\n");
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        fcg_gpu_memref_free_f16(d_A);
        fcg_gpu_memref_free_f16(d_B);
        fcg_gpu_memref_free_f16(d_C);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_add_f16(d_A, d_B, d_C);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_add_f16(d_A, d_B, d_C);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f16(h_A_mem);
    fcg_memref_free_f16(h_B_mem);
    fcg_memref_free_f16(h_C_gpu_mem);
    fcg_memref_free_f16(h_C_cpu_mem);
    fcg_gpu_memref_free_f16(d_A);
    fcg_gpu_memref_free_f16(d_B);
    fcg_gpu_memref_free_f16(d_C);
    return gflops;
}

static double bench_gpu_relu_f16(int M, int N, int repeat) {
    struct MemRef h_X_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_Y_gpu_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_Y_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!h_X_mem.aligned || !h_Y_gpu_mem.aligned || !h_Y_cpu_mem.aligned) {
        fcg_memref_free_f16(h_X_mem);
        fcg_memref_free_f16(h_Y_gpu_mem);
        fcg_memref_free_f16(h_Y_cpu_mem);
        return 0.0;
    }
    _Float16* h_X = (_Float16*)h_X_mem.aligned;
    _Float16* h_Y_gpu = (_Float16*)h_Y_gpu_mem.aligned;
    _Float16* h_Y_cpu = (_Float16*)h_Y_cpu_mem.aligned;

    fill_random_f16(h_X, M * N);
    memset(h_Y_gpu, 0, M * N * sizeof(_Float16));
    memset(h_Y_cpu, 0, M * N * sizeof(_Float16));

    struct MemRef d_X = fcg_gpu_memref_alloc_f16(M, N);
    struct MemRef d_Y = fcg_gpu_memref_alloc_f16(M, N);
    if (!d_X.aligned || !d_Y.aligned) {
        fcg_memref_free_f16(h_X_mem);
        fcg_memref_free_f16(h_Y_gpu_mem);
        fcg_memref_free_f16(h_Y_cpu_mem);
        fcg_gpu_memref_free_f16(d_X);
        fcg_gpu_memref_free_f16(d_Y);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f16(h_X_mem, d_X);

    relu_cpu_f16(h_X, h_Y_cpu, M, N);
    fcg_relu_f16(d_X, d_Y);
    fcg_gpu_memref_copy_d2h_f16(d_Y, h_Y_gpu_mem);

    if (!check_matrices_f16(h_Y_cpu, h_Y_gpu, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for relu_f16 (GPU)\n");
        fcg_memref_free_f16(h_X_mem);
        fcg_memref_free_f16(h_Y_gpu_mem);
        fcg_memref_free_f16(h_Y_cpu_mem);
        fcg_gpu_memref_free_f16(d_X);
        fcg_gpu_memref_free_f16(d_Y);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_relu_f16(d_X, d_Y);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_relu_f16(d_X, d_Y);
    double elapsed = get_time_sec() - start;

    double flops = (double)M * N * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f16(h_X_mem);
    fcg_memref_free_f16(h_Y_gpu_mem);
    fcg_memref_free_f16(h_Y_cpu_mem);
    fcg_gpu_memref_free_f16(d_X);
    fcg_gpu_memref_free_f16(d_Y);
    return gflops;
}

static double bench_gpu_full_flow_f16(int M, int N, int K, int repeat) {
    struct MemRef h_A_mem = fcg_memref_alloc_f16(M, K);
    struct MemRef h_B_mem = fcg_memref_alloc_f16(K, N);
    struct MemRef h_bias_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_C_gpu_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_C_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!h_A_mem.aligned || !h_B_mem.aligned || !h_bias_mem.aligned ||
        !h_C_gpu_mem.aligned || !h_C_cpu_mem.aligned) {
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_bias_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        return 0.0;
    }
    _Float16* h_A = (_Float16*)h_A_mem.aligned;
    _Float16* h_B = (_Float16*)h_B_mem.aligned;
    _Float16* h_bias = (_Float16*)h_bias_mem.aligned;
    _Float16* h_C_gpu = (_Float16*)h_C_gpu_mem.aligned;
    _Float16* h_C_cpu = (_Float16*)h_C_cpu_mem.aligned;

    fill_random_f16(h_A, M * K);
    fill_random_f16(h_B, K * N);
    fill_random_f16(h_bias, M * N);
    memset(h_C_gpu, 0, M * N * sizeof(_Float16));
    memset(h_C_cpu, 0, M * N * sizeof(_Float16));

    struct MemRef d_A = fcg_gpu_memref_alloc_f16(M, K);
    struct MemRef d_B = fcg_gpu_memref_alloc_f16(K, N);
    struct MemRef d_bias = fcg_gpu_memref_alloc_f16(M, N);
    struct MemRef d_C = fcg_gpu_memref_alloc_f16(M, N);
    if (!d_A.aligned || !d_B.aligned || !d_bias.aligned || !d_C.aligned) {
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_bias_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        fcg_gpu_memref_free_f16(d_A);
        fcg_gpu_memref_free_f16(d_B);
        fcg_gpu_memref_free_f16(d_bias);
        fcg_gpu_memref_free_f16(d_C);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f16(h_A_mem, d_A);
    fcg_gpu_memref_copy_h2d_f16(h_B_mem, d_B);
    fcg_gpu_memref_copy_h2d_f16(h_bias_mem, d_bias);

    full_flow_cpu_f16(h_A, h_B, h_bias, h_C_cpu, M, N, K);
    fcg_full_flow_f16(d_A, d_B, d_bias, d_C);
    fcg_gpu_memref_copy_d2h_f16(d_C, h_C_gpu_mem);

    if (!check_matrices_f16(h_C_cpu, h_C_gpu, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for full_flow_f16 (GPU)\n");
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_bias_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        fcg_gpu_memref_free_f16(d_A);
        fcg_gpu_memref_free_f16(d_B);
        fcg_gpu_memref_free_f16(d_bias);
        fcg_gpu_memref_free_f16(d_C);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_full_flow_f16(d_A, d_B, d_bias, d_C);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_full_flow_f16(d_A, d_B, d_bias, d_C);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f16(h_A_mem);
    fcg_memref_free_f16(h_B_mem);
    fcg_memref_free_f16(h_bias_mem);
    fcg_memref_free_f16(h_C_gpu_mem);
    fcg_memref_free_f16(h_C_cpu_mem);
    fcg_gpu_memref_free_f16(d_A);
    fcg_gpu_memref_free_f16(d_B);
    fcg_gpu_memref_free_f16(d_bias);
    fcg_gpu_memref_free_f16(d_C);
    return gflops;
}

static double bench_gpu_fusion_full_flow_f16(int M, int N, int K, int repeat) {
    struct MemRef h_A_mem = fcg_memref_alloc_f16(M, K);
    struct MemRef h_B_mem = fcg_memref_alloc_f16(K, N);
    struct MemRef h_bias_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_C_gpu_mem = fcg_memref_alloc_f16(M, N);
    struct MemRef h_C_cpu_mem = fcg_memref_alloc_f16(M, N);
    if (!h_A_mem.aligned || !h_B_mem.aligned || !h_bias_mem.aligned ||
        !h_C_gpu_mem.aligned || !h_C_cpu_mem.aligned) {
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_bias_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        return 0.0;
    }
    _Float16* h_A = (_Float16*)h_A_mem.aligned;
    _Float16* h_B = (_Float16*)h_B_mem.aligned;
    _Float16* h_bias = (_Float16*)h_bias_mem.aligned;
    _Float16* h_C_gpu = (_Float16*)h_C_gpu_mem.aligned;
    _Float16* h_C_cpu = (_Float16*)h_C_cpu_mem.aligned;

    fill_random_f16(h_A, M * K);
    fill_random_f16(h_B, K * N);
    fill_random_f16(h_bias, M * N);
    memset(h_C_gpu, 0, M * N * sizeof(_Float16));
    memset(h_C_cpu, 0, M * N * sizeof(_Float16));

    struct MemRef d_A = fcg_gpu_memref_alloc_f16(M, K);
    struct MemRef d_B = fcg_gpu_memref_alloc_f16(K, N);
    struct MemRef d_bias = fcg_gpu_memref_alloc_f16(M, N);
    struct MemRef d_C = fcg_gpu_memref_alloc_f16(M, N);
    if (!d_A.aligned || !d_B.aligned || !d_bias.aligned || !d_C.aligned) {
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_bias_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        fcg_gpu_memref_free_f16(d_A);
        fcg_gpu_memref_free_f16(d_B);
        fcg_gpu_memref_free_f16(d_bias);
        fcg_gpu_memref_free_f16(d_C);
        return 0.0;
    }

    fcg_gpu_memref_copy_h2d_f16(h_A_mem, d_A);
    fcg_gpu_memref_copy_h2d_f16(h_B_mem, d_B);
    fcg_gpu_memref_copy_h2d_f16(h_bias_mem, d_bias);

    full_flow_cpu_f16(h_A, h_B, h_bias, h_C_cpu, M, N, K);
    fcg_fusion_full_flow_f16(d_A, d_B, d_bias, d_C);
    fcg_gpu_memref_copy_d2h_f16(d_C, h_C_gpu_mem);

    if (!check_matrices_f16(h_C_cpu, h_C_gpu, M, N, EPSILON_F16)) {
        fprintf(stderr, "Correctness check FAILED for fusion_full_flow_f16 (GPU)\n");
        fcg_memref_free_f16(h_A_mem);
        fcg_memref_free_f16(h_B_mem);
        fcg_memref_free_f16(h_bias_mem);
        fcg_memref_free_f16(h_C_gpu_mem);
        fcg_memref_free_f16(h_C_cpu_mem);
        fcg_gpu_memref_free_f16(d_A);
        fcg_gpu_memref_free_f16(d_B);
        fcg_gpu_memref_free_f16(d_bias);
        fcg_gpu_memref_free_f16(d_C);
        exit(1);
    }

    for (int i = 0; i < 2; ++i) fcg_fusion_full_flow_f16(d_A, d_B, d_bias, d_C);
    double start = get_time_sec();
    for (int i = 0; i < repeat; ++i) fcg_fusion_full_flow_f16(d_A, d_B, d_bias, d_C);
    double elapsed = get_time_sec() - start;

    double flops = (2.0 * M * N * K + M * N + M * N) * repeat;
    double gflops = flops / (elapsed * 1e9);

    fcg_memref_free_f16(h_A_mem);
    fcg_memref_free_f16(h_B_mem);
    fcg_memref_free_f16(h_bias_mem);
    fcg_memref_free_f16(h_C_gpu_mem);
    fcg_memref_free_f16(h_C_cpu_mem);
    fcg_gpu_memref_free_f16(d_A);
    fcg_gpu_memref_free_f16(d_B);
    fcg_gpu_memref_free_f16(d_bias);
    fcg_gpu_memref_free_f16(d_C);
    return gflops;
}

int main(int argc, char **argv) {
    int M = 128, N = 128, K = 128;
    int repeat = 1;

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

    printf("=== Performance Comparison (CPU vs GPU) ===\n");
    printf("MLIR Pass: %s\n", pass_name);
    printf("M=%d, N=%d, K=%d, repeat=%d\n\n", M, N, K, repeat);

    printf("---- f32 ----\n");
    double cpu, gpu;

    cpu = bench_cpu_matmul_f32(M, N, K, repeat);
    gpu = bench_gpu_matmul_f32(M, N, K, repeat);
    printf("matmul      : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    cpu = bench_cpu_add_f32(M, N, repeat);
    gpu = bench_gpu_add_f32(M, N, repeat);
    printf("add         : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    cpu = bench_cpu_relu_f32(M, N, repeat);
    gpu = bench_gpu_relu_f32(M, N, repeat);
    printf("relu        : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    cpu = bench_cpu_full_flow_f32(M, N, K, repeat);
    gpu = bench_gpu_full_flow_f32(M, N, K, repeat);
    printf("full_flow   : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    gpu = bench_gpu_fusion_full_flow_f32(M, N, K, repeat);
    printf("fusion_flow : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    printf("\n---- f16 ----\n");

    cpu = bench_cpu_matmul_f16(M, N, K, repeat);
    gpu = bench_gpu_matmul_f16(M, N, K, repeat);
    printf("matmul      : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    cpu = bench_cpu_add_f16(M, N, repeat);
    gpu = bench_gpu_add_f16(M, N, repeat);
    printf("add         : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    cpu = bench_cpu_relu_f16(M, N, repeat);
    gpu = bench_gpu_relu_f16(M, N, repeat);
    printf("relu        : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    cpu = bench_cpu_full_flow_f16(M, N, K, repeat);
    gpu = bench_gpu_full_flow_f16(M, N, K, repeat);
    printf("full_flow   : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    gpu = bench_gpu_fusion_full_flow_f16(M, N, K, repeat);
    printf("fusion_flow : CPU = %8.2f GFLOPS, GPU = %8.2f GFLOPS, ratio = %.2fx\n",
           cpu, gpu, cpu / gpu);

    return 0;
}
