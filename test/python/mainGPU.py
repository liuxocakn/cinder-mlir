#!/usr/bin/env python3

import argparse
import sys
import os
import time
import random
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'binding', 'python'))
from fcg_ffi_gpu import (
    fcg_load_library,
    fcg_unload_library,
    fcg_matmul_f32,
    fcg_matmul_f16,
    fcg_add_f32,
    fcg_add_f16,
    fcg_relu_f32,
    fcg_relu_f16,
    fcg_full_flow_f32,
    fcg_full_flow_f16,
    fcg_fusion_full_flow_f32,
    fcg_fusion_full_flow_f16,
    fcg_memref_alloc_f32,
    fcg_memref_free_f32,
    fcg_memref_zero_f32,
    fcg_memref_from_list_f32,
    fcg_memref_to_list_f32,
    fcg_memref_alloc_f16,
    fcg_memref_free_f16,
    fcg_memref_zero_f16,
    fcg_memref_from_list_f16,
    fcg_memref_to_list_f16,
    fcg_gpu_memref_alloc_f32,
    fcg_gpu_memref_free_f32,
    fcg_gpu_memref_alloc_f16,
    fcg_gpu_memref_free_f16,
    fcg_gpu_memref_copy_h2d_f32,
    fcg_gpu_memref_copy_d2h_f32,
    fcg_gpu_memref_copy_h2d_f16,
    fcg_gpu_memref_copy_d2h_f16,
)

EPSILON_F32 = 1e-5
EPSILON_F16 = 1

def matmul_cpu(A, B):
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    C = [[0.0 for _ in range(N)] for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def add_cpu(A, B):
    M = len(A)
    N = len(A[0])
    C = [[0.0 for _ in range(N)] for _ in range(M)]
    for i in range(M):
        for j in range(N):
            C[i][j] = A[i][j] + B[i][j]
    return C

def relu_cpu(X):
    M = len(X)
    N = len(X[0])
    Y = [[0.0 for _ in range(N)] for _ in range(M)]
    for i in range(M):
        for j in range(N):
            x = X[i][j]
            Y[i][j] = x if x > 0.0 else 0.0
    return Y

def full_flow_cpu(A, B, bias):
    C = matmul_cpu(A, B)
    M = len(C)
    N = len(C[0])
    for i in range(M):
        for j in range(N):
            val = C[i][j] + bias[i][j]
            C[i][j] = val if val > 0.0 else 0.0
    return C

def random_matrix_f32(rows, cols):
    return [[random.random() * 2.0 - 1.0 for _ in range(cols)] for _ in range(rows)]

def random_matrix_f16(rows, cols):
    return [[random.random() * 2.0 - 1.0 for _ in range(cols)] for _ in range(rows)]

def check_matrices(cpu, mlir, name, eps):
    M = len(cpu)
    N = len(cpu[0])
    max_diff = 0.0
    max_i = max_j = 0
    for i in range(M):
        for j in range(N):
            diff = abs(cpu[i][j] - mlir[i][j])
            if diff > max_diff:
                max_diff = diff
                max_i, max_j = i, j
    if max_diff > eps:
        print(f"Correctness check FAILED for {name}")
        print(f"  First mismatch at [{max_i}][{max_j}]: CPU={cpu[max_i][max_j]:.6f}, MLIR={mlir[max_i][max_j]:.6f}, diff={max_diff:.6e}, eps={eps:.6e}")
        sys.exit(1)

def bench_cpu_matmul_f32(M, N, K, repeat):
    A = random_matrix_f32(M, K)
    B = random_matrix_f32(K, N)
    for _ in range(2):
        matmul_cpu(A, B)
    start = time.perf_counter()
    for _ in range(repeat):
        C_cpu = matmul_cpu(A, B)
    elapsed = time.perf_counter() - start
    flops = 2.0 * M * N * K * repeat
    gflops = flops / (elapsed * 1e9)
    return gflops, C_cpu

def bench_mlir_matmul_f32(M, N, K, repeat):
    A_list = random_matrix_f32(M, K)
    B_list = random_matrix_f32(K, N)

    A_cpu = fcg_memref_alloc_f32(M, K)
    B_cpu = fcg_memref_alloc_f32(K, N)
    fcg_memref_from_list_f32(A_cpu, A_list)
    fcg_memref_from_list_f32(B_cpu, B_list)

    A_gpu = fcg_gpu_memref_alloc_f32(M, K)
    B_gpu = fcg_gpu_memref_alloc_f32(K, N)
    out_gpu = fcg_gpu_memref_alloc_f32(M, N)

    fcg_gpu_memref_copy_h2d_f32(A_cpu, A_gpu)
    fcg_gpu_memref_copy_h2d_f32(B_cpu, B_gpu)

    fcg_memref_free_f32(A_cpu)
    fcg_memref_free_f32(B_cpu)

    out_cpu_check = fcg_memref_alloc_f32(M, N)
    fcg_matmul_f32(A_gpu, B_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f32(out_gpu, out_cpu_check)
    C_mlir = fcg_memref_to_list_f32(out_cpu_check)
    C_cpu = matmul_cpu(A_list, B_list)
    check_matrices(C_cpu, C_mlir, "matmul_f32", EPSILON_F32)
    fcg_memref_free_f32(out_cpu_check)

    for _ in range(2):
        fcg_matmul_f32(A_gpu, B_gpu, out_gpu)

    start = time.perf_counter()
    for _ in range(repeat):
        fcg_matmul_f32(A_gpu, B_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = 2.0 * M * N * K * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f32(A_gpu)
    fcg_gpu_memref_free_f32(B_gpu)
    fcg_gpu_memref_free_f32(out_gpu)

    return gflops, C_cpu


def bench_cpu_add_f32(M, N, repeat):
    A = random_matrix_f32(M, N)
    B = random_matrix_f32(M, N)
    for _ in range(2):
        add_cpu(A, B)
    start = time.perf_counter()
    for _ in range(repeat):
        C_cpu = add_cpu(A, B)
    elapsed = time.perf_counter() - start
    flops = M * N * repeat
    gflops = flops / (elapsed * 1e9)
    return gflops, C_cpu

def bench_mlir_add_f32(M, N, repeat):
    A_list = random_matrix_f32(M, N)
    B_list = random_matrix_f32(M, N)

    A_cpu = fcg_memref_alloc_f32(M, N)
    B_cpu = fcg_memref_alloc_f32(M, N)
    fcg_memref_from_list_f32(A_cpu, A_list)
    fcg_memref_from_list_f32(B_cpu, B_list)

    A_gpu = fcg_gpu_memref_alloc_f32(M, N)
    B_gpu = fcg_gpu_memref_alloc_f32(M, N)
    out_gpu = fcg_gpu_memref_alloc_f32(M, N)

    fcg_gpu_memref_copy_h2d_f32(A_cpu, A_gpu)
    fcg_gpu_memref_copy_h2d_f32(B_cpu, B_gpu)

    fcg_memref_free_f32(A_cpu)
    fcg_memref_free_f32(B_cpu)

    out_cpu_check = fcg_memref_alloc_f32(M, N)
    fcg_add_f32(A_gpu, B_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f32(out_gpu, out_cpu_check)
    C_mlir = fcg_memref_to_list_f32(out_cpu_check)
    C_cpu = add_cpu(A_list, B_list)
    check_matrices(C_cpu, C_mlir, "add_f32", EPSILON_F32)
    fcg_memref_free_f32(out_cpu_check)

    for _ in range(2):
        fcg_add_f32(A_gpu, B_gpu, out_gpu)
    start = time.perf_counter()
    for _ in range(repeat):
        fcg_add_f32(A_gpu, B_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = M * N * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f32(A_gpu)
    fcg_gpu_memref_free_f32(B_gpu)
    fcg_gpu_memref_free_f32(out_gpu)

    return gflops, C_cpu


def bench_cpu_relu_f32(M, N, repeat):
    X = random_matrix_f32(M, N)
    for _ in range(2):
        relu_cpu(X)
    start = time.perf_counter()
    for _ in range(repeat):
        Y_cpu = relu_cpu(X)
    elapsed = time.perf_counter() - start
    flops = M * N * repeat
    gflops = flops / (elapsed * 1e9)
    return gflops, Y_cpu

def bench_mlir_relu_f32(M, N, repeat):
    X_list = random_matrix_f32(M, N)

    X_cpu = fcg_memref_alloc_f32(M, N)
    fcg_memref_from_list_f32(X_cpu, X_list)

    X_gpu = fcg_gpu_memref_alloc_f32(M, N)
    out_gpu = fcg_gpu_memref_alloc_f32(M, N)

    fcg_gpu_memref_copy_h2d_f32(X_cpu, X_gpu)
    fcg_memref_free_f32(X_cpu)

    out_cpu_check = fcg_memref_alloc_f32(M, N)
    fcg_relu_f32(X_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f32(out_gpu, out_cpu_check)
    Y_mlir = fcg_memref_to_list_f32(out_cpu_check)
    Y_cpu = relu_cpu(X_list)
    check_matrices(Y_cpu, Y_mlir, "relu_f32", EPSILON_F32)
    fcg_memref_free_f32(out_cpu_check)

    for _ in range(2):
        fcg_relu_f32(X_gpu, out_gpu)
    start = time.perf_counter()
    for _ in range(repeat):
        fcg_relu_f32(X_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = M * N * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f32(X_gpu)
    fcg_gpu_memref_free_f32(out_gpu)

    return gflops, Y_cpu


def bench_cpu_full_flow_f32(M, N, K, repeat):
    A = random_matrix_f32(M, K)
    B = random_matrix_f32(K, N)
    bias = random_matrix_f32(M, N)
    for _ in range(2):
        full_flow_cpu(A, B, bias)
    start = time.perf_counter()
    for _ in range(repeat):
        C_cpu = full_flow_cpu(A, B, bias)
    elapsed = time.perf_counter() - start
    flops = (2.0 * M * N * K + M * N + M * N) * repeat
    gflops = flops / (elapsed * 1e9)
    return gflops, C_cpu

def bench_mlir_full_flow_f32(M, N, K, repeat):
    A_list = random_matrix_f32(M, K)
    B_list = random_matrix_f32(K, N)
    bias_list = random_matrix_f32(M, N)

    A_cpu = fcg_memref_alloc_f32(M, K)
    B_cpu = fcg_memref_alloc_f32(K, N)
    bias_cpu = fcg_memref_alloc_f32(M, N)
    fcg_memref_from_list_f32(A_cpu, A_list)
    fcg_memref_from_list_f32(B_cpu, B_list)
    fcg_memref_from_list_f32(bias_cpu, bias_list)

    A_gpu = fcg_gpu_memref_alloc_f32(M, K)
    B_gpu = fcg_gpu_memref_alloc_f32(K, N)
    bias_gpu = fcg_gpu_memref_alloc_f32(M, N)
    out_gpu = fcg_gpu_memref_alloc_f32(M, N)

    fcg_gpu_memref_copy_h2d_f32(A_cpu, A_gpu)
    fcg_gpu_memref_copy_h2d_f32(B_cpu, B_gpu)
    fcg_gpu_memref_copy_h2d_f32(bias_cpu, bias_gpu)

    fcg_memref_free_f32(A_cpu)
    fcg_memref_free_f32(B_cpu)
    fcg_memref_free_f32(bias_cpu)

    out_cpu_check = fcg_memref_alloc_f32(M, N)
    fcg_full_flow_f32(A_gpu, B_gpu, bias_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f32(out_gpu, out_cpu_check)
    C_mlir = fcg_memref_to_list_f32(out_cpu_check)
    C_cpu = full_flow_cpu(A_list, B_list, bias_list)
    check_matrices(C_cpu, C_mlir, "full_flow_f32", EPSILON_F32)
    fcg_memref_free_f32(out_cpu_check)

    for _ in range(2):
        fcg_full_flow_f32(A_gpu, B_gpu, bias_gpu, out_gpu)
    start = time.perf_counter()
    for _ in range(repeat):
        fcg_full_flow_f32(A_gpu, B_gpu, bias_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = (2.0 * M * N * K + M * N + M * N) * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f32(A_gpu)
    fcg_gpu_memref_free_f32(B_gpu)
    fcg_gpu_memref_free_f32(bias_gpu)
    fcg_gpu_memref_free_f32(out_gpu)

    return gflops, C_cpu

def bench_mlir_fusion_full_flow_f32(M, N, K, repeat):
    A_list = random_matrix_f32(M, K)
    B_list = random_matrix_f32(K, N)
    bias_list = random_matrix_f32(M, N)

    A_cpu = fcg_memref_alloc_f32(M, K)
    B_cpu = fcg_memref_alloc_f32(K, N)
    bias_cpu = fcg_memref_alloc_f32(M, N)
    fcg_memref_from_list_f32(A_cpu, A_list)
    fcg_memref_from_list_f32(B_cpu, B_list)
    fcg_memref_from_list_f32(bias_cpu, bias_list)

    A_gpu = fcg_gpu_memref_alloc_f32(M, K)
    B_gpu = fcg_gpu_memref_alloc_f32(K, N)
    bias_gpu = fcg_gpu_memref_alloc_f32(M, N)
    out_gpu = fcg_gpu_memref_alloc_f32(M, N)

    fcg_gpu_memref_copy_h2d_f32(A_cpu, A_gpu)
    fcg_gpu_memref_copy_h2d_f32(B_cpu, B_gpu)
    fcg_gpu_memref_copy_h2d_f32(bias_cpu, bias_gpu)

    fcg_memref_free_f32(A_cpu)
    fcg_memref_free_f32(B_cpu)
    fcg_memref_free_f32(bias_cpu)

    out_cpu_check = fcg_memref_alloc_f32(M, N)
    fcg_fusion_full_flow_f32(A_gpu, B_gpu, bias_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f32(out_gpu, out_cpu_check)
    C_mlir = fcg_memref_to_list_f32(out_cpu_check)
    C_cpu = full_flow_cpu(A_list, B_list, bias_list)
    check_matrices(C_cpu, C_mlir, "fusion_full_flow_f32", EPSILON_F32)
    fcg_memref_free_f32(out_cpu_check)

    for _ in range(2):
        fcg_fusion_full_flow_f32(A_gpu, B_gpu, bias_gpu, out_gpu)
    start = time.perf_counter()
    for _ in range(repeat):
        fcg_fusion_full_flow_f32(A_gpu, B_gpu, bias_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = (2.0 * M * N * K + M * N + M * N) * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f32(A_gpu)
    fcg_gpu_memref_free_f32(B_gpu)
    fcg_gpu_memref_free_f32(bias_gpu)
    fcg_gpu_memref_free_f32(out_gpu)

    return gflops, C_cpu

def bench_cpu_matmul_f16(M, N, K, repeat):
    A = random_matrix_f16(M, K)
    B = random_matrix_f16(K, N)
    for _ in range(2):
        matmul_cpu(A, B)
    start = time.perf_counter()
    for _ in range(repeat):
        C_cpu = matmul_cpu(A, B)
    elapsed = time.perf_counter() - start
    flops = 2.0 * M * N * K * repeat
    gflops = flops / (elapsed * 1e9)
    return gflops, C_cpu

def bench_mlir_matmul_f16(M, N, K, repeat):
    A_list = random_matrix_f16(M, K)
    B_list = random_matrix_f16(K, N)

    A_cpu = fcg_memref_alloc_f16(M, K)
    B_cpu = fcg_memref_alloc_f16(K, N)
    fcg_memref_from_list_f16(A_cpu, A_list)
    fcg_memref_from_list_f16(B_cpu, B_list)

    A_gpu = fcg_gpu_memref_alloc_f16(M, K)
    B_gpu = fcg_gpu_memref_alloc_f16(K, N)
    out_gpu = fcg_gpu_memref_alloc_f16(M, N)

    fcg_gpu_memref_copy_h2d_f16(A_cpu, A_gpu)
    fcg_gpu_memref_copy_h2d_f16(B_cpu, B_gpu)

    fcg_memref_free_f16(A_cpu)
    fcg_memref_free_f16(B_cpu)

    out_cpu_check = fcg_memref_alloc_f16(M, N)
    fcg_matmul_f16(A_gpu, B_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f16(out_gpu, out_cpu_check)
    C_mlir = fcg_memref_to_list_f16(out_cpu_check)
    C_cpu = matmul_cpu(A_list, B_list)
    check_matrices(C_cpu, C_mlir, "matmul_f16", EPSILON_F16)
    fcg_memref_free_f16(out_cpu_check)

    for _ in range(2):
        fcg_matmul_f16(A_gpu, B_gpu, out_gpu)
    start = time.perf_counter()
    for _ in range(repeat):
        fcg_matmul_f16(A_gpu, B_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = 2.0 * M * N * K * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f16(A_gpu)
    fcg_gpu_memref_free_f16(B_gpu)
    fcg_gpu_memref_free_f16(out_gpu)

    return gflops, C_cpu


def bench_cpu_add_f16(M, N, repeat):
    A = random_matrix_f16(M, N)
    B = random_matrix_f16(M, N)
    for _ in range(2):
        add_cpu(A, B)
    start = time.perf_counter()
    for _ in range(repeat):
        C_cpu = add_cpu(A, B)
    elapsed = time.perf_counter() - start
    flops = M * N * repeat
    gflops = flops / (elapsed * 1e9)
    return gflops, C_cpu

def bench_mlir_add_f16(M, N, repeat):
    A_list = random_matrix_f16(M, N)
    B_list = random_matrix_f16(M, N)

    A_cpu = fcg_memref_alloc_f16(M, N)
    B_cpu = fcg_memref_alloc_f16(M, N)
    fcg_memref_from_list_f16(A_cpu, A_list)
    fcg_memref_from_list_f16(B_cpu, B_list)

    A_gpu = fcg_gpu_memref_alloc_f16(M, N)
    B_gpu = fcg_gpu_memref_alloc_f16(M, N)
    out_gpu = fcg_gpu_memref_alloc_f16(M, N)

    fcg_gpu_memref_copy_h2d_f16(A_cpu, A_gpu)
    fcg_gpu_memref_copy_h2d_f16(B_cpu, B_gpu)

    fcg_memref_free_f16(A_cpu)
    fcg_memref_free_f16(B_cpu)

    out_cpu_check = fcg_memref_alloc_f16(M, N)
    fcg_add_f16(A_gpu, B_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f16(out_gpu, out_cpu_check)
    C_mlir = fcg_memref_to_list_f16(out_cpu_check)
    C_cpu = add_cpu(A_list, B_list)
    check_matrices(C_cpu, C_mlir, "add_f16", EPSILON_F16)
    fcg_memref_free_f16(out_cpu_check)

    for _ in range(2):
        fcg_add_f16(A_gpu, B_gpu, out_gpu)
    start = time.perf_counter()
    for _ in range(repeat):
        fcg_add_f16(A_gpu, B_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = M * N * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f16(A_gpu)
    fcg_gpu_memref_free_f16(B_gpu)
    fcg_gpu_memref_free_f16(out_gpu)

    return gflops, C_cpu


def bench_cpu_relu_f16(M, N, repeat):
    X = random_matrix_f16(M, N)
    for _ in range(2):
        relu_cpu(X)
    start = time.perf_counter()
    for _ in range(repeat):
        Y_cpu = relu_cpu(X)
    elapsed = time.perf_counter() - start
    flops = M * N * repeat
    gflops = flops / (elapsed * 1e9)
    return gflops, Y_cpu

def bench_mlir_relu_f16(M, N, repeat):
    X_list = random_matrix_f16(M, N)

    X_cpu = fcg_memref_alloc_f16(M, N)
    fcg_memref_from_list_f16(X_cpu, X_list)

    X_gpu = fcg_gpu_memref_alloc_f16(M, N)
    out_gpu = fcg_gpu_memref_alloc_f16(M, N)

    fcg_gpu_memref_copy_h2d_f16(X_cpu, X_gpu)
    fcg_memref_free_f16(X_cpu)

    out_cpu_check = fcg_memref_alloc_f16(M, N)
    fcg_relu_f16(X_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f16(out_gpu, out_cpu_check)
    Y_mlir = fcg_memref_to_list_f16(out_cpu_check)
    Y_cpu = relu_cpu(X_list)
    check_matrices(Y_cpu, Y_mlir, "relu_f16", EPSILON_F16)
    fcg_memref_free_f16(out_cpu_check)

    for _ in range(2):
        fcg_relu_f16(X_gpu, out_gpu)
    start = time.perf_counter()
    for _ in range(repeat):
        fcg_relu_f16(X_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = M * N * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f16(X_gpu)
    fcg_gpu_memref_free_f16(out_gpu)

    return gflops, Y_cpu


def bench_cpu_full_flow_f16(M, N, K, repeat):
    A = random_matrix_f16(M, K)
    B = random_matrix_f16(K, N)
    bias = random_matrix_f16(M, N)
    for _ in range(2):
        full_flow_cpu(A, B, bias)
    start = time.perf_counter()
    for _ in range(repeat):
        C_cpu = full_flow_cpu(A, B, bias)
    elapsed = time.perf_counter() - start
    flops = (2.0 * M * N * K + M * N + M * N) * repeat
    gflops = flops / (elapsed * 1e9)
    return gflops, C_cpu

def bench_mlir_full_flow_f16(M, N, K, repeat):
    A_list = random_matrix_f16(M, K)
    B_list = random_matrix_f16(K, N)
    bias_list = random_matrix_f16(M, N)

    A_cpu = fcg_memref_alloc_f16(M, K)
    B_cpu = fcg_memref_alloc_f16(K, N)
    bias_cpu = fcg_memref_alloc_f16(M, N)
    fcg_memref_from_list_f16(A_cpu, A_list)
    fcg_memref_from_list_f16(B_cpu, B_list)
    fcg_memref_from_list_f16(bias_cpu, bias_list)

    A_gpu = fcg_gpu_memref_alloc_f16(M, K)
    B_gpu = fcg_gpu_memref_alloc_f16(K, N)
    bias_gpu = fcg_gpu_memref_alloc_f16(M, N)
    out_gpu = fcg_gpu_memref_alloc_f16(M, N)

    fcg_gpu_memref_copy_h2d_f16(A_cpu, A_gpu)
    fcg_gpu_memref_copy_h2d_f16(B_cpu, B_gpu)
    fcg_gpu_memref_copy_h2d_f16(bias_cpu, bias_gpu)

    fcg_memref_free_f16(A_cpu)
    fcg_memref_free_f16(B_cpu)
    fcg_memref_free_f16(bias_cpu)

    out_cpu_check = fcg_memref_alloc_f16(M, N)
    fcg_full_flow_f16(A_gpu, B_gpu, bias_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f16(out_gpu, out_cpu_check)
    C_mlir = fcg_memref_to_list_f16(out_cpu_check)
    C_cpu = full_flow_cpu(A_list, B_list, bias_list)
    check_matrices(C_cpu, C_mlir, "full_flow_f16", EPSILON_F16)
    fcg_memref_free_f16(out_cpu_check)

    for _ in range(2):
        fcg_full_flow_f16(A_gpu, B_gpu, bias_gpu, out_gpu)
    start = time.perf_counter()
    for _ in range(repeat):
        fcg_full_flow_f16(A_gpu, B_gpu, bias_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = (2.0 * M * N * K + M * N + M * N) * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f16(A_gpu)
    fcg_gpu_memref_free_f16(B_gpu)
    fcg_gpu_memref_free_f16(bias_gpu)
    fcg_gpu_memref_free_f16(out_gpu)

    return gflops, C_cpu

def bench_mlir_fusion_full_flow_f16(M, N, K, repeat):
    A_list = random_matrix_f16(M, K)
    B_list = random_matrix_f16(K, N)
    bias_list = random_matrix_f16(M, N)

    A_cpu = fcg_memref_alloc_f16(M, K)
    B_cpu = fcg_memref_alloc_f16(K, N)
    bias_cpu = fcg_memref_alloc_f16(M, N)
    fcg_memref_from_list_f16(A_cpu, A_list)
    fcg_memref_from_list_f16(B_cpu, B_list)
    fcg_memref_from_list_f16(bias_cpu, bias_list)

    A_gpu = fcg_gpu_memref_alloc_f16(M, K)
    B_gpu = fcg_gpu_memref_alloc_f16(K, N)
    bias_gpu = fcg_gpu_memref_alloc_f16(M, N)
    out_gpu = fcg_gpu_memref_alloc_f16(M, N)

    fcg_gpu_memref_copy_h2d_f16(A_cpu, A_gpu)
    fcg_gpu_memref_copy_h2d_f16(B_cpu, B_gpu)
    fcg_gpu_memref_copy_h2d_f16(bias_cpu, bias_gpu)

    fcg_memref_free_f16(A_cpu)
    fcg_memref_free_f16(B_cpu)
    fcg_memref_free_f16(bias_cpu)

    out_cpu_check = fcg_memref_alloc_f16(M, N)
    fcg_fusion_full_flow_f16(A_gpu, B_gpu, bias_gpu, out_gpu)
    fcg_gpu_memref_copy_d2h_f16(out_gpu, out_cpu_check)
    C_mlir = fcg_memref_to_list_f16(out_cpu_check)
    C_cpu = full_flow_cpu(A_list, B_list, bias_list)
    check_matrices(C_cpu, C_mlir, "fusion_full_flow_f16", EPSILON_F16)
    fcg_memref_free_f16(out_cpu_check)

    for _ in range(2):
        fcg_fusion_full_flow_f16(A_gpu, B_gpu, bias_gpu, out_gpu)
    start = time.perf_counter()
    for _ in range(repeat):
        fcg_fusion_full_flow_f16(A_gpu, B_gpu, bias_gpu, out_gpu)
    elapsed = time.perf_counter() - start

    flops = (2.0 * M * N * K + M * N + M * N) * repeat
    gflops = flops / (elapsed * 1e9)

    fcg_gpu_memref_free_f16(A_gpu)
    fcg_gpu_memref_free_f16(B_gpu)
    fcg_gpu_memref_free_f16(bias_gpu)
    fcg_gpu_memref_free_f16(out_gpu)

    return gflops, C_cpu

def main():
    parser = argparse.ArgumentParser(description='Performance Comparison: CPU vs MLIR (GPU)')
    parser.add_argument('M', type=int, nargs='?', default=128, help='Matrix dimension M (default: 128)')
    parser.add_argument('N', type=int, nargs='?', default=128, help='Matrix dimension N (default: 128)')
    parser.add_argument('K', type=int, nargs='?', default=128, help='Matrix dimension K (default: 128)')
    parser.add_argument('repeat', type=int, nargs='?', default=5, help='Number of repetitions (default: 5)')
    parser.add_argument('--pass', dest='pass_name', default='Baseline',
        help='MLIR pass to use: specify a library name (e.g., Baseline, Vectorize, or custom .so) (default: Baseline)')

    args = parser.parse_args()

    M, N, K, repeat = args.M, args.N, args.K, args.repeat

    pass_arg = args.pass_name
    if pass_arg.endswith('.so'):
        lib_filename = pass_arg
        if lib_filename.startswith('libFCGPass') and lib_filename.endswith('.so'):
            short_name = lib_filename[10:-3]
        else:
            short_name = lib_filename
    else:
        lib_filename = f'libFCGPass{pass_arg}.so'
        short_name = pass_arg

    lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'test', 'lib', lib_filename)
    try:
        fcg_load_library(lib_path)
    except Exception as e:
        print(f"Failed to load library: {e}")
        return

    print("=== Performance Comparison (CPU vs MLIR/Python) ===")
    print(f"MLIR Pass: {short_name}")
    print(f"M={M}, N={N}, K={K}, repeat={repeat}\n")

    print("---- f32 ----")
    g_cpu, _ = bench_cpu_matmul_f32(M, N, K, repeat)
    g_mlir, _ = bench_mlir_matmul_f32(M, N, K, repeat)
    print(f"matmul      : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir:8.2f} GFLOPS, ratio = {g_cpu/g_mlir:.2f}x")

    g_cpu, _ = bench_cpu_add_f32(M, N, repeat)
    g_mlir, _ = bench_mlir_add_f32(M, N, repeat)
    print(f"add         : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir:8.2f} GFLOPS, ratio = {g_cpu/g_mlir:.2f}x")

    g_cpu, _ = bench_cpu_relu_f32(M, N, repeat)
    g_mlir, _ = bench_mlir_relu_f32(M, N, repeat)
    print(f"relu        : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir:8.2f} GFLOPS, ratio = {g_cpu/g_mlir:.2f}x")

    g_cpu, _ = bench_cpu_full_flow_f32(M, N, K, repeat)
    g_mlir, _ = bench_mlir_full_flow_f32(M, N, K, repeat)
    print(f"full_flow   : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir:8.2f} GFLOPS, ratio = {g_cpu/g_mlir:.2f}x")

    g_mlir_fusion, _ = bench_mlir_fusion_full_flow_f32(M, N, K, repeat)
    print(f"fusion_flow : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir_fusion:8.2f} GFLOPS, ratio = {g_cpu/g_mlir_fusion:.2f}x")

    print("\n---- f16 ----")
    g_cpu, _ = bench_cpu_matmul_f16(M, N, K, repeat)
    g_mlir, _ = bench_mlir_matmul_f16(M, N, K, repeat)
    print(f"matmul      : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir:8.2f} GFLOPS, ratio = {g_cpu/g_mlir:.2f}x")

    g_cpu, _ = bench_cpu_add_f16(M, N, repeat)
    g_mlir, _ = bench_mlir_add_f16(M, N, repeat)
    print(f"add         : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir:8.2f} GFLOPS, ratio = {g_cpu/g_mlir:.2f}x")

    g_cpu, _ = bench_cpu_relu_f16(M, N, repeat)
    g_mlir, _ = bench_mlir_relu_f16(M, N, repeat)
    print(f"relu        : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir:8.2f} GFLOPS, ratio = {g_cpu/g_mlir:.2f}x")

    g_cpu, _ = bench_cpu_full_flow_f16(M, N, K, repeat)
    g_mlir, _ = bench_mlir_full_flow_f16(M, N, K, repeat)
    print(f"full_flow   : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir:8.2f} GFLOPS, ratio = {g_cpu/g_mlir:.2f}x")

    g_mlir_fusion, _ = bench_mlir_fusion_full_flow_f16(M, N, K, repeat)
    print(f"fusion_flow : CPU = {g_cpu:8.2f} GFLOPS, MLIR = {g_mlir_fusion:8.2f} GFLOPS, ratio = {g_cpu/g_mlir_fusion:.2f}x")

    fcg_unload_library()

if __name__ == "__main__":
    main()
