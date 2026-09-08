#!/usr/bin/env python3

from .ops import (
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
)

from ._utils import (
    fcg_memref_alloc_f32,
    fcg_memref_free_f32,
    fcg_memref_zero_f32,
    fcg_memref_alloc_f16,
    fcg_memref_free_f16,
    fcg_memref_zero_f16,
    fcg_memref_from_list_f32,
    fcg_memref_to_list_f32,
    fcg_memref_from_list_f16,
    fcg_memref_to_list_f16,
    fcg_gpu_memref_alloc_f32,
    fcg_gpu_memref_free_f32,
    fcg_gpu_memref_alloc_f16,
    fcg_gpu_memref_free_f16,
    fcg_gpu_memref_copy_h2d_f32,
    fcg_gpu_memref_copy_d2h_f32,
    fcg_gpu_memref_copy_h2d_f16,
    fcg_gpu_memref_copy_d2h_f16
)

from ._core import fcg_load_library, fcg_unload_library

__version__ = "0.1.0"

__all__ = [
    "fcg_load_library",
    "fcg_unload_library",

    "fcg_matmul_f32",
    "fcg_matmul_f16",
    "fcg_add_f32",
    "fcg_add_f16",
    "fcg_relu_f32",
    "fcg_relu_f16",
    "fcg_full_flow_f32",
    "fcg_full_flow_f16",
    "fcg_fusion_full_flow_f32",
    "fcg_fusion_full_flow_f16",

    "fcg_memref_alloc_f32",
    "fcg_memref_free_f32",
    "fcg_memref_zero_f32",
    "fcg_memref_alloc_f16",
    "fcg_memref_free_f16",
    "fcg_memref_zero_f16",

    "fcg_memref_from_list_f32",
    "fcg_memref_to_list_f32",
    "fcg_memref_from_list_f16",
    "fcg_memref_to_list_f16",
]
