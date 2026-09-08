#!/usr/bin/env python3

from . import _core as core


def _unwrap_memref(m, name="memref"):
    if not isinstance(m, core.MemRef):
        raise TypeError(f"{name} must be a MemRef, got {type(m)}")
    return (m.data, m.aligned, m.offset,
            m.sizes[0], m.sizes[1], m.strides[0], m.strides[1])


def fcg_matmul_f32(A, B, out):
    A_args = _unwrap_memref(A, "A")
    B_args = _unwrap_memref(B, "B")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_matmul_f32(*A_args, *B_args, *out_args)


def fcg_matmul_f16(A, B, out):
    A_args = _unwrap_memref(A, "A")
    B_args = _unwrap_memref(B, "B")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_matmul_f16(*A_args, *B_args, *out_args)


def fcg_add_f32(A, B, out):
    A_args = _unwrap_memref(A, "A")
    B_args = _unwrap_memref(B, "B")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_add_f32(*A_args, *B_args, *out_args)


def fcg_add_f16(A, B, out):
    A_args = _unwrap_memref(A, "A")
    B_args = _unwrap_memref(B, "B")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_add_f16(*A_args, *B_args, *out_args)


def fcg_relu_f32(input, out):
    input_args = _unwrap_memref(input, "input")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_relu_f32(*input_args, *out_args)


def fcg_relu_f16(input, out):
    input_args = _unwrap_memref(input, "input")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_relu_f16(*input_args, *out_args)


def fcg_full_flow_f32(A, B, bias, out):
    A_args = _unwrap_memref(A, "A")
    B_args = _unwrap_memref(B, "B")
    bias_args = _unwrap_memref(bias, "bias")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_full_flow_f32(*A_args, *B_args, *bias_args, *out_args)


def fcg_full_flow_f16(A, B, bias, out):
    A_args = _unwrap_memref(A, "A")
    B_args = _unwrap_memref(B, "B")
    bias_args = _unwrap_memref(bias, "bias")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_full_flow_f16(*A_args, *B_args, *bias_args, *out_args)


def fcg_fusion_full_flow_f32(A, B, bias, out):
    A_args = _unwrap_memref(A, "A")
    B_args = _unwrap_memref(B, "B")
    bias_args = _unwrap_memref(bias, "bias")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_fusion_full_flow_f32(*A_args, *B_args, *bias_args, *out_args)


def fcg_fusion_full_flow_f16(A, B, bias, out):
    A_args = _unwrap_memref(A, "A")
    B_args = _unwrap_memref(B, "B")
    bias_args = _unwrap_memref(bias, "bias")
    out_args = _unwrap_memref(out, "out")
    return core._lib.test_fusion_full_flow_f16(*A_args, *B_args, *bias_args, *out_args)
