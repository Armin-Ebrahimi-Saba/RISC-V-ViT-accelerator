/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Depth-Anything V2 Small inference engine for the CV32E40P.
 *
 * Numeric format
 * --------------
 * Weights   : per-output-channel symmetric int8 (prepared by export_dav2.py)
 * Activations: int16 restricted to +-DAV2_ACT_QMAX (14 bit) with one float
 *              scale per tensor, recomputed dynamically from the actual data.
 * Accumulate : int32. The 14-bit activation limit guarantees that the longest
 *              reduction in the network (K = 1536) cannot overflow:
 *              1536 * 8191 * 127 = 1.60e9 < 2^31.
 *
 * Floating point is used only for O(channels) bookkeeping (deriving scales and
 * requantisation multipliers). Every per-element inner loop is pure integer, so
 * the lack of an FPU on the CV32E40P costs almost nothing.
 *
 * Memory layout
 * -------------
 * Feature maps are channel-last (NHWC): a tensor is an (n x c) matrix whose
 * rows are pixels/tokens and whose columns are channels. This makes im2col
 * rows contiguous and keeps DDR3 bursts sequential, which matters because the
 * DDR3 last-level cache is direct-mapped.
 */

#ifndef DAV2_H
#define DAV2_H

#include <stddef.h>
#include <stdint.h>

#define DAV2_ACT_QMAX  8191   /* 14-bit activation magnitude   */
#define DAV2_QK_QMAX   2047   /* q/k magnitude inside attention */
#define DAV2_W_QMAX     127

#define DAV2_EMBED_DIM 384
#define DAV2_N_HEADS     6
#define DAV2_HEAD_DIM   64
#define DAV2_N_BLOCKS   12
#define DAV2_PATCH      14
#define DAV2_FEATURES   64

/* ------------------------------------------------------------------ blob */

#define DAV2_MAGIC    0x32564144u   /* 'DAV2' */
#define DAV2_VERSION  2u
#define DAV2_NAME_LEN 40

enum { DAV2_DT_F32 = 0, DAV2_DT_I8 = 1, DAV2_DT_I16 = 2, DAV2_DT_I32 = 3 };

typedef struct {
    uint32_t magic, version, n_tensors, dir_off, data_off, total_bytes;
    uint32_t reserved[6];
} dav2_header_t;

typedef struct {
    char     name[DAV2_NAME_LEN];
    uint32_t dtype, nbytes, offset, dims[4], pad;
} dav2_record_t;

/* A quantised weight matrix: (m x k) int8 rows, one float scale per row and an
 * optional float bias per row. */
typedef struct {
    const int8_t  *w;
    const float   *s;
    const float   *b;   /* NULL when the layer has no bias */
    int            m, k;
} dav2_qw_t;

/* An activation tensor: n rows of c int16 channels, real = v * scale.
 *
 * amax_q is the largest |v| in the tensor when the producer knows it exactly
 * (every operator that streams its output past once does, and the
 * accelerator reports it for requantised results), or -1 when it does not.
 * Consumers that need the range -- the residual add -- use it instead of a
 * scan; a scan and a known amax_q give identical results. */
typedef struct {
    int16_t *v;
    float    scale;
    int      n, c;
    int32_t  amax_q;
} dav2_tensor_t;

/* ------------------------------------------------------------- blob access */

void        dav2_blob_init(const void *blob);
const void *dav2_find(const char *name, uint32_t *nbytes);
/* Looks up "<base>.w"/".s"/".b" and fills in a weight descriptor. */
void        dav2_qw(dav2_qw_t *out, const char *base, int expect_k);

/* ---------------------------------------------------------------- arena */

void   dav2_arena_init(void *base, size_t bytes);
void  *dav2_arena_alloc(size_t bytes);
size_t dav2_arena_mark(void);
void   dav2_arena_release(size_t mark);
size_t dav2_arena_peak(void);
/* Non-zero if any allocation has failed since dav2_arena_init. */
extern int dav2_arena_failed;

/* Split a positive float into (mult, shift) with mult in [2^30, 2^31) so that
 * x*m can be evaluated as (x*mult) >> shift in integer arithmetic. */
void dav2_make_multiplier(float m, int32_t *mult, int *shift);

/* --------------------------------------------------------------- tensors */

dav2_tensor_t dav2_tensor_new(int n, int c);
/* Quantise a float buffer into a fresh activation tensor. */
void dav2_quantize_f32(const float *src, int n, int c, dav2_tensor_t *out);

/* -------------------------------------------------------------------- ops */

/* out = A * W^T + bias, with dynamic output scale. out must be preallocated
 * with n = A->n and c = W->m. */
void dav2_qgemm(const dav2_tensor_t *a, const dav2_qw_t *w, dav2_tensor_t *out);
/* dav2_qgemm, then dav2_add(res, result) if res, then dav2_relu if relu --
 * bit-identical, but on the accelerator the add and the ReLU run inside the
 * requantisation job. out may be res (in-place residual update). */
void dav2_qgemm_ex(const dav2_tensor_t *a, const dav2_qw_t *wt,
                   const dav2_tensor_t *res, int relu, dav2_tensor_t *out);

/* LayerNorm over channels, producing a fresh dynamic scale. */
void dav2_layernorm(const dav2_tensor_t *in, const float *g, const float *b,
                    dav2_tensor_t *out);

/* out = a + b (elementwise, different input scales allowed). */
void dav2_add(const dav2_tensor_t *a, const dav2_tensor_t *b, dav2_tensor_t *out);

void dav2_relu(dav2_tensor_t *t);
void dav2_gelu(dav2_tensor_t *t);

/* NHWC bilinear resize with align_corners=true, matching F.interpolate. */
void dav2_interpolate(const dav2_tensor_t *in, int h, int w,
                      int oh, int ow, dav2_tensor_t *out);

/* im2col for an NHWC feature map; produces (oh*ow) rows of (kh*kw*c). */
void dav2_im2col(const dav2_tensor_t *in, int h, int w,
                 int kh, int kw, int stride, int pad, dav2_tensor_t *cols);

/* Convenience: im2col + qgemm, returning an (oh*ow x m) tensor. */
dav2_tensor_t dav2_conv2d(const dav2_tensor_t *in, int h, int w,
                          const dav2_qw_t *wt, int k, int stride, int pad,
                          int *oh, int *ow);
dav2_tensor_t dav2_conv2d_ex(const dav2_tensor_t *in, int h, int w,
                             const dav2_qw_t *wt, int k, int stride, int pad,
                             const dav2_tensor_t *res, int relu,
                             int *oh_out, int *ow_out);

/* Non-overlapping transposed convolution (kernel == stride). */
dav2_tensor_t dav2_conv_transpose(const dav2_tensor_t *in, int h, int w,
                                  const dav2_qw_t *wt, int stride,
                                  int *oh, int *ow);

/* ------------------------------------------------------------------ model */

typedef struct {
    int   size;      /* input side length      */
    int   grid;      /* patches per side       */
    int   n_tokens;  /* grid*grid + 1          */
} dav2_cfg_t;

/* Runs the full network. depth_out receives grid*14 x grid*14 float values
 * (the same units as the PyTorch model's output). */
void dav2_infer(const dav2_cfg_t *cfg, float *depth_out);

/* Input image.
 *
 * The picture used to travel inside the weight blob as two tensors, which
 * meant a new image cost a full 25 MB weight transfer. It is now supplied
 * separately: the caller points the engine at a quantised int16 HWC buffer
 * (size x size x 3, values in -8191..8191) and its scale, and can do so again
 * before every dav2_infer without touching the weights. On the board the
 * buffer lives at a fixed DDR3 address the host writes over JTAG; the host
 * build passes a malloc'd array. */
void dav2_set_image(const int16_t *hwc, float scale);

/* On-chip scratch buffer and word-wise copies -- see dav2_ops.c. 40 kB:
 * enough for one attention head's q, k and transposed v (3 x 82 x 64 int16)
 * plus its score row, with room for a 16-row requantisation tile. */
#define DAV2_SCRATCH_ELEMS 20480
extern int16_t dav2_scratch[DAV2_SCRATCH_ELEMS];
void dav2_copy16(int16_t *dst, const int16_t *src, size_t n);
void dav2_zero16(int16_t *dst, size_t n);

/* Progress hook, implemented by the caller (prints to stdout on the SoC). */
void dav2_progress(const char *stage);

/* Free-running cycle (or time) counter, implemented by the caller: mcycle on
 * the SoC, a wall clock on the host. Only differences are used. */
uint64_t dav2_cycles(void);

/* Where the time goes. Every operator adds its own cycles to one bucket, so
 * after a frame the split between the accelerated matmuls and the CPU work
 * around them can be read off directly instead of estimated. */
enum {
    DAV2_PROF_GEMM_ACCEL,   /* accelerator running, CPU waiting            */
    DAV2_PROF_GEMM_CPU,     /* software matmul (only when no accelerator)  */
    DAV2_PROF_REQUANT,      /* int32 accumulators -> int16 activations     */
    DAV2_PROF_ATTENTION,    /* QK^T, softmax, PV per head (CPU)            */
    DAV2_PROF_LAYERNORM,
    DAV2_PROF_GELU,
    DAV2_PROF_ADD_RELU,     /* residual adds, relu                         */
    DAV2_PROF_IM2COL,       /* conv patch gather                           */
    DAV2_PROF_INTERP,       /* bilinear resize / transpose conv            */
    DAV2_PROF_OTHER,        /* everything not above (copies, glue)         */
    DAV2_PROF_N
};
extern const char *const dav2_prof_name[DAV2_PROF_N];
void     dav2_prof_reset(void);
void     dav2_prof_add(int bucket, uint64_t cycles);
uint64_t dav2_prof_get(int bucket);
void     dav2_prof_report(uint64_t frame_cycles);   /* prints the table */

#endif /* DAV2_H */
