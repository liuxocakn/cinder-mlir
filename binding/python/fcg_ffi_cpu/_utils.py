#!/usr/bin/env python3

import ctypes
import struct
from ctypes import c_float, c_uint16, POINTER, memset

from . import _core as core


def fcg_memref_alloc_f32(rows, cols):
    return core._lib.test_memref_alloc_f32(rows, cols)

def fcg_memref_free_f32(memref):
    if memref.data:
        core._lib.test_memref_free_f32(
            memref.data,
            memref.aligned,
            memref.offset,
            memref.sizes[0],
            memref.sizes[1],
            memref.strides[0],
            memref.strides[1]
        )

def fcg_memref_zero_f32(memref):
    rows, cols = memref.sizes[0], memref.sizes[1]
    size = rows * cols * ctypes.sizeof(c_float)
    memset(memref.aligned, 0, size)


def fcg_memref_alloc_f16(rows, cols):
    return core._lib.test_memref_alloc_f16(rows, cols)

def fcg_memref_free_f16(memref):
    if memref.data:
        core._lib.test_memref_free_f16(
            memref.data,
            memref.aligned,
            memref.offset,
            memref.sizes[0],
            memref.sizes[1],
            memref.strides[0],
            memref.strides[1]
        )

def fcg_memref_zero_f16(memref):
    rows, cols = memref.sizes[0], memref.sizes[1]
    size = rows * cols * ctypes.sizeof(c_uint16)
    memset(memref.aligned, 0, size)


def fcg_memref_from_list_f32(memref, listdata):
    rows, cols = memref.sizes[0], memref.sizes[1]
    if len(listdata) != rows or (rows > 0 and len(listdata[0]) != cols):
        raise ValueError(f"Data shape {len(listdata)}x{len(listdata[0])} doesn't match MemRef {rows}x{cols}")
    ptr = ctypes.cast(memref.aligned, POINTER(c_float))
    for i in range(rows):
        row_data = listdata[i]
        base = i * memref.strides[0]
        for j in range(cols):
            ptr[base + j * memref.strides[1]] = float(row_data[j])

def fcg_memref_to_list_f32(memref):
    rows, cols = memref.sizes[0], memref.sizes[1]
    ptr = ctypes.cast(memref.aligned, POINTER(c_float))
    result = []
    for i in range(rows):
        row = []
        base = i * memref.strides[0]
        for j in range(cols):
            row.append(ptr[base + j * memref.strides[1]])
        result.append(row)
    return result


def fcg_memref_from_list_f16(memref, listdata):
    rows, cols = memref.sizes[0], memref.sizes[1]
    if len(listdata) != rows or (rows > 0 and len(listdata[0]) != cols):
        raise ValueError(f"Data shape {len(listdata)}x{len(listdata[0])} doesn't match MemRef {rows}x{cols}")
    ptr = ctypes.cast(memref.aligned, POINTER(c_uint16))
    for i in range(rows):
        row_data = listdata[i]
        base = i * memref.strides[0]
        for j in range(cols):
            val = float(row_data[j])
            packed = struct.pack('e', val)
            uint16_val = int.from_bytes(packed, 'little')
            ptr[base + j * memref.strides[1]] = uint16_val

def fcg_memref_to_list_f16(memref):
    rows, cols = memref.sizes[0], memref.sizes[1]
    ptr = ctypes.cast(memref.aligned, POINTER(c_uint16))
    result = []
    for i in range(rows):
        row = []
        base = i * memref.strides[0]
        for j in range(cols):
            uint16_val = ptr[base + j * memref.strides[1]]
            packed = uint16_val.to_bytes(2, 'little')
            val = struct.unpack('e', packed)[0]
            row.append(val)
        result.append(row)
    return result
