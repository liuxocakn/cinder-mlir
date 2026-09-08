#ifndef __FCG_EXPORT_CPU_H__
#define __FCG_EXPORT_CPU_H__

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MemRef {
    float* data;
    float* aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

static inline struct MemRef fcg_make_memref(void* data, int64_t rows, int64_t cols) {
    struct MemRef m;
    m.data = (float*)data;
    m.aligned = (float*)data;
    m.offset = 0;
    m.sizes[0] = rows;
    m.sizes[1] = cols;
    m.strides[0] = cols;
    m.strides[1] = 1;
    return m;
}

struct MemRef test_matmul_f32(
    float* A_data, float* A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    float* B_data, float* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    float* out_data, float* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_matmul_f16(
    void* A_data, void* A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    void* B_data, void* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    void* out_data, void* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_add_f32(
    float* A_data, float* A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    float* B_data, float* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    float* out_data, float* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_add_f16(
    void* A_data, void* A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    void* B_data, void* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    void* out_data, void* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_relu_f32(
    float* input_data, float* input_aligned, int64_t input_offset,
    int64_t input_size0, int64_t input_size1, int64_t input_stride0, int64_t input_stride1,
    float* out_data, float* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_relu_f16(
    void* input_data, void* input_aligned, int64_t input_offset,
    int64_t input_size0, int64_t input_size1, int64_t input_stride0, int64_t input_stride1,
    void* out_data, void* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_full_flow_f32(
    float* A_data, float* A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    float* B_data, float* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    float* bias_data, float* bias_aligned, int64_t bias_offset,
    int64_t bias_size0, int64_t bias_size1, int64_t bias_stride0, int64_t bias_stride1,
    float* out_data, float* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_full_flow_f16(
    void* A_data, void* A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    void* B_data, void* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    void* bias_data, void* bias_aligned, int64_t bias_offset,
    int64_t bias_size0, int64_t bias_size1, int64_t bias_stride0, int64_t bias_stride1,
    void* out_data, void* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_fusion_full_flow_f32(
    float* A_data, float* A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    float* B_data, float* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    float* bias_data, float* bias_aligned, int64_t bias_offset,
    int64_t bias_size0, int64_t bias_size1, int64_t bias_stride0, int64_t bias_stride1,
    float* out_data, float* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_fusion_full_flow_f16(
    void* A_data, void* A_aligned, int64_t A_offset,
    int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    void* B_data, void* B_aligned, int64_t B_offset,
    int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    void* bias_data, void* bias_aligned, int64_t bias_offset,
    int64_t bias_size0, int64_t bias_size1, int64_t bias_stride0, int64_t bias_stride1,
    void* out_data, void* out_aligned, int64_t out_offset,
    int64_t out_size0, int64_t out_size1, int64_t out_stride0, int64_t out_stride1
);

struct MemRef test_memref_alloc_f32(int64_t x, int64_t y);
void test_memref_free_f32(
    void* data, void* aligned, int64_t offset,
    int64_t size0, int64_t size1, int64_t stride0, int64_t stride1
);

struct MemRef test_memref_alloc_f16(int64_t x, int64_t y);
void test_memref_free_f16(
    void* data, void* aligned, int64_t offset,
    int64_t size0, int64_t size1, int64_t stride0, int64_t stride1
);

static inline struct MemRef fcg_matmul_f32(struct MemRef A, struct MemRef B, struct MemRef out) {
    return test_matmul_f32(
        A.data, A.aligned, A.offset,
        A.sizes[0], A.sizes[1], A.strides[0], A.strides[1],
        B.data, B.aligned, B.offset,
        B.sizes[0], B.sizes[1], B.strides[0], B.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_matmul_f16(struct MemRef A, struct MemRef B, struct MemRef out) {
    return test_matmul_f16(
        A.data, A.aligned, A.offset,
        A.sizes[0], A.sizes[1], A.strides[0], A.strides[1],
        B.data, B.aligned, B.offset,
        B.sizes[0], B.sizes[1], B.strides[0], B.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_add_f32(struct MemRef A, struct MemRef B, struct MemRef out) {
    return test_add_f32(
        A.data, A.aligned, A.offset,
        A.sizes[0], A.sizes[1], A.strides[0], A.strides[1],
        B.data, B.aligned, B.offset,
        B.sizes[0], B.sizes[1], B.strides[0], B.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_add_f16(struct MemRef A, struct MemRef B, struct MemRef out) {
    return test_add_f16(
        A.data, A.aligned, A.offset,
        A.sizes[0], A.sizes[1], A.strides[0], A.strides[1],
        B.data, B.aligned, B.offset,
        B.sizes[0], B.sizes[1], B.strides[0], B.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_relu_f32(struct MemRef input, struct MemRef out) {
    return test_relu_f32(
        input.data, input.aligned, input.offset,
        input.sizes[0], input.sizes[1], input.strides[0], input.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_relu_f16(struct MemRef input, struct MemRef out) {
    return test_relu_f16(
        input.data, input.aligned, input.offset,
        input.sizes[0], input.sizes[1], input.strides[0], input.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_full_flow_f32(struct MemRef A, struct MemRef B, struct MemRef bias, struct MemRef out) {
    return test_full_flow_f32(
        A.data, A.aligned, A.offset,
        A.sizes[0], A.sizes[1], A.strides[0], A.strides[1],
        B.data, B.aligned, B.offset,
        B.sizes[0], B.sizes[1], B.strides[0], B.strides[1],
        bias.data, bias.aligned, bias.offset,
        bias.sizes[0], bias.sizes[1], bias.strides[0], bias.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_full_flow_f16(struct MemRef A, struct MemRef B, struct MemRef bias, struct MemRef out) {
    return test_full_flow_f16(
        A.data, A.aligned, A.offset,
        A.sizes[0], A.sizes[1], A.strides[0], A.strides[1],
        B.data, B.aligned, B.offset,
        B.sizes[0], B.sizes[1], B.strides[0], B.strides[1],
        bias.data, bias.aligned, bias.offset,
        bias.sizes[0], bias.sizes[1], bias.strides[0], bias.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_fusion_full_flow_f32(struct MemRef A, struct MemRef B, struct MemRef bias, struct MemRef out) {
    return test_fusion_full_flow_f32(
        A.data, A.aligned, A.offset,
        A.sizes[0], A.sizes[1], A.strides[0], A.strides[1],
        B.data, B.aligned, B.offset,
        B.sizes[0], B.sizes[1], B.strides[0], B.strides[1],
        bias.data, bias.aligned, bias.offset,
        bias.sizes[0], bias.sizes[1], bias.strides[0], bias.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_fusion_full_flow_f16(struct MemRef A, struct MemRef B, struct MemRef bias, struct MemRef out) {
    return test_fusion_full_flow_f16(
        A.data, A.aligned, A.offset,
        A.sizes[0], A.sizes[1], A.strides[0], A.strides[1],
        B.data, B.aligned, B.offset,
        B.sizes[0], B.sizes[1], B.strides[0], B.strides[1],
        bias.data, bias.aligned, bias.offset,
        bias.sizes[0], bias.sizes[1], bias.strides[0], bias.strides[1],
        out.data, out.aligned, out.offset,
        out.sizes[0], out.sizes[1], out.strides[0], out.strides[1]
    );
}

static inline struct MemRef fcg_memref_alloc_f32(int64_t x, int64_t y) {
    return test_memref_alloc_f32(x, y);
}

static inline void fcg_memref_free_f32(struct MemRef memref) {
    test_memref_free_f32(
        memref.data,
        memref.aligned,
        memref.offset,
        memref.sizes[0],
        memref.sizes[1],
        memref.strides[0],
        memref.strides[1]
    );
}

static inline struct MemRef fcg_memref_alloc_f16(int64_t x, int64_t y) {
    return test_memref_alloc_f16(x, y);
}

static inline void fcg_memref_free_f16(struct MemRef memref) {
    test_memref_free_f16(
        memref.data,
        memref.aligned,
        memref.offset,
        memref.sizes[0],
        memref.sizes[1],
        memref.strides[0],
        memref.strides[1]
    );
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __FCG_EXPORT_CPU_H__
