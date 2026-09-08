#!/usr/bin/env python3

import os
import ctypes
from ctypes import c_float, c_int64, c_void_p, Structure, POINTER

_lib = None
_lib_path = None

class MemRef(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("aligned", c_void_p),
        ("offset", c_int64),
        ("sizes", c_int64 * 2),
        ("strides", c_int64 * 2),
    ]

    def __repr__(self):
        data_addr = hex(self.data) if self.data else "0x0"
        aligned_addr = hex(self.aligned) if self.aligned else "0x0"
        return (f"MemRef (\n"
                f"  data={data_addr},\n"
                f"  aligned={aligned_addr},\n"
                f"  offset={self.offset},\n"
                f"  sizes=({self.sizes[0]}, {self.sizes[1]}),\n"
                f"  strides=({self.strides[0]}, {self.strides[1]})\n"
                f")")


def fcg_load_library(path):
    global _lib, _lib_path

    if _lib is not None:
        fcg_unload_library()

    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shared library not found: {path}")

    try:
        _lib = ctypes.CDLL(path)
    except OSError as e:
        raise OSError(f"Failed to load library {path}: {e}")

    _lib_path = path

    memref_argtypes = [c_void_p, c_void_p, c_int64, c_int64, c_int64, c_int64, c_int64]
    memref1 = memref_argtypes
    memref2 = memref_argtypes + memref_argtypes
    memref3 = memref_argtypes + memref_argtypes + memref_argtypes
    memref4 = memref_argtypes + memref_argtypes + memref_argtypes + memref_argtypes

    _lib.test_matmul_f32.argtypes           = memref3
    _lib.test_matmul_f32.restype            = MemRef

    _lib.test_matmul_f16.argtypes           = memref3
    _lib.test_matmul_f16.restype            = MemRef

    _lib.test_add_f32.argtypes              = memref3
    _lib.test_add_f32.restype               = MemRef

    _lib.test_add_f16.argtypes              = memref3
    _lib.test_add_f16.restype               = MemRef

    _lib.test_relu_f32.argtypes             = memref2
    _lib.test_relu_f32.restype              = MemRef

    _lib.test_relu_f16.argtypes             = memref2
    _lib.test_relu_f16.restype              = MemRef

    _lib.test_full_flow_f32.argtypes        = memref4
    _lib.test_full_flow_f32.restype         = MemRef

    _lib.test_full_flow_f16.argtypes        = memref4
    _lib.test_full_flow_f16.restype         = MemRef

    _lib.test_fusion_full_flow_f32.argtypes = memref4
    _lib.test_fusion_full_flow_f32.restype  = MemRef

    _lib.test_fusion_full_flow_f16.argtypes = memref4
    _lib.test_fusion_full_flow_f16.restype  = MemRef

    _lib.test_memref_alloc_f32.argtypes     = [c_int64, c_int64]
    _lib.test_memref_alloc_f32.restype      = MemRef

    _lib.test_memref_free_f32.argtypes      = memref1
    _lib.test_memref_free_f32.restype       = None

    _lib.test_memref_alloc_f16.argtypes     = [c_int64, c_int64]
    _lib.test_memref_alloc_f16.restype      = MemRef

    _lib.test_memref_free_f16.argtypes      = memref1
    _lib.test_memref_free_f16.restype       = None

    _lib.test_gpu_memref_alloc_f32.argtypes = [c_int64, c_int64]
    _lib.test_gpu_memref_alloc_f32.restype  = MemRef

    _lib.test_gpu_memref_alloc_f16.argtypes = [c_int64, c_int64]
    _lib.test_gpu_memref_alloc_f16.restype  = MemRef

    _lib.test_gpu_memref_free_f32.argtypes  = memref1
    _lib.test_gpu_memref_free_f32.restype   = None

    _lib.test_gpu_memref_free_f16.argtypes  = memref1
    _lib.test_gpu_memref_free_f16.restype   = None

    _lib.test_gpu_memref_copy_h2d_f32.argtypes = memref2
    _lib.test_gpu_memref_copy_h2d_f32.restype  = None

    _lib.test_gpu_memref_copy_d2h_f32.argtypes = memref2
    _lib.test_gpu_memref_copy_d2h_f32.restype  = None

    _lib.test_gpu_memref_copy_h2d_f16.argtypes = memref2
    _lib.test_gpu_memref_copy_h2d_f16.restype  = None

    _lib.test_gpu_memref_copy_d2h_f16.argtypes = memref2
    _lib.test_gpu_memref_copy_d2h_f16.restype  = None

    return _lib

def fcg_unload_library():
    global _lib, _lib_path
    _lib = None
    _lib_path = None
