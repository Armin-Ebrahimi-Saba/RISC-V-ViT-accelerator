/* SPDX-License-Identifier: CC0-1.0
 * SPDX-FileCopyrightText: 2026 RVLab Student Project
 *
 * Depth-Anything V2 Small inference engine for the CV32E40P.
 *
 * Numeric format
 * --------------
 * Weights   : per-output-channel symmetric int8 (prepared by export_dav2.py)
 * Activations: int16 restricted to +-DAV2_ACT_QMAX (14 bit) with one scale
 *              per tensor, recomputed dynamically from the actual data.
 * Accumulate : int32. The 14-bit activation limit guarantees that the longest
 *              reduction in the network (K = 1536) cannot overflow:
 *              1536 * 8191 * 127 = 1.60e9 < 2^31.
 *
 * There is no floating point. The CV32E40P has no FPU, and each float
 * operation would be a libgcc call of 35 to 200 cycles. A scale is a pair
 * of integers, value = m * 2^-sh (dav2_xf.h). The small float parameters of
 * the model (row scales, biases, LayerNorm gamma and beta, embeddings) are
 * converted to integer forms offline by tools/dav2_blob_int.py.
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

#include "dav2_xf.h"

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
#define DAV2_VERSION  4u   /* 4: integer parameters, final norm folded (tools/dav2_blob_int.py) */
#define DAV2_NAME_LEN 40

enum { DAV2_DT_F32 = 0, DAV2_DT_I8 = 1, DAV2_DT_I16 = 2, DAV2_DT_I32 = 3 };

/* Integer formats of the parameters in a version 4 blob. */
#define DAV2_POS_Q        24   /* cls_token, pos_embed: value = int / 2^24     */
#define DAV2_LN_G_Q       15   /* LayerNorm gamma                              */
#define DAV2_LN_B_Q       16   /* LayerNorm beta                               */
#define DAV2_PHI_Z0       (-8) /* gelu_phi: Phi(z), z = -8 + i/128, i=0..2048 */
#define DAV2_PHI_STEP_LOG2 7
#define DAV2_PHI_Q        30

typedef struct {
    uint32_t magic, version, n_tensors, dir_off, data_off, total_bytes;
    uint32_t reserved[6];
} dav2_header_t;

typedef struct {
    char     name[DAV2_NAME_LEN];
    uint32_t dtype, nbytes, offset, dims[4], pad;
} dav2_record_t;

/* A quantised weight matrix: (m x k) int8 rows, one scale per row and an
 * optional bias per row. Scale and bias are (m, sh) integer pairs. */
typedef struct {
    const int8_t    *w;
    const dav2_xf_t *s;
    const dav2_xf_t *b;   /* NULL when the layer has no bias */
    int              m, k;
} dav2_qw_t;

/* An activation tensor: n rows of c int16 channels, real = v * scale.
 *
 * amax_q is the largest |v| in the tensor when the producer knows it exactly
 * (every operator that streams its output past once does, and the
 * accelerator reports it for requantised results), or -1 when it does not.
 * Consumers that need the range -- the residual add -- use it instead of a
 * scan; a scan and a known amax_q give identical results. */
typedef struct {
    int16_t   *v;
    dav2_xf_t  scale;
    int        n, c;
    int32_t    amax_q;
    /* Each row's largest and smallest value, one word per row (max in bits
     * 31:16, min in bits 15:0), or NULL when unknown. Set by the residual
     * updates from the requantisation job's row statistics (CTRL.ostats);
     * LayerNorm uses them instead of finding the extremes itself. */
    uint32_t  *rst;
    /* Each row's sum and sum of squares, three words per row (sum, then the
     * sum of squares, bits 31:0 and 63:32), or NULL when unknown. Set with
     * rst when the block reports them (CTRL.osums); LayerNorm then needs no
     * pass over its input. */
    uint32_t  *rsum;
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


/* --------------------------------------------------------------- tensors */

dav2_tensor_t dav2_tensor_new(int n, int c);
/* The token matrix: row 0 = cls + pos[0], row p+1 = patch[p] + pos[p+1],
 * with cls and pos in Q24 (DAV2_POS_Q). out must hold (n_patch + 1) x c. */
void dav2_embed_tokens(const dav2_tensor_t *patches, const int32_t *cls_q24,
                       const int32_t *pos_q24, dav2_tensor_t *out);

/* -------------------------------------------------------------------- ops */

/* out = A * W^T + bias, with dynamic output scale. out must be preallocated
 * with n = A->n and c = W->m. */
void dav2_qgemm(const dav2_tensor_t *a, const dav2_qw_t *w, dav2_tensor_t *out);
/* dav2_qgemm and each output column's largest and smallest value (exact,
 * from the accumulators' extremes; no pass over the output) */
void dav2_qgemm_colext(const dav2_tensor_t *a, const dav2_qw_t *w, dav2_tensor_t *out,
                       int16_t *cmax, int16_t *cmin);
/* dav2_qgemm, then dav2_add(res, result) if res, then dav2_relu if relu --
 * bit-identical, but on the accelerator the add and the ReLU run inside the
 * requantisation job. out may be res (in-place residual update). */
void dav2_qgemm_ex(const dav2_tensor_t *a, const dav2_qw_t *wt,
                   const dav2_tensor_t *res, int relu, dav2_tensor_t *out);

/* dav2_qgemm then dav2_gelu, bit-identical. On the accelerator the GELU
 * table is applied inside the requantisation job (CTRL.lut). */
void dav2_qgemm_gelu(const dav2_tensor_t *a, const dav2_qw_t *w, dav2_tensor_t *out);

/* LayerNorm over channels, producing a fresh dynamic scale. g_q15 = NULL
 * means no gamma and beta: out = (in - mean) / std. That is the final norm,
 * whose gamma and beta are folded offline into proj0..3
 * (tools/dav2_blob_int.py). */
void dav2_layernorm(const dav2_tensor_t *in, const int32_t *g_q15,
                    const int32_t *b_q16, dav2_tensor_t *out);

/* out = a + b (elementwise, different input scales allowed). */
void dav2_add(const dav2_tensor_t *a, const dav2_tensor_t *b, dav2_tensor_t *out);

void dav2_relu(dav2_tensor_t *t);
/* dst = relu(src), in one pass (dst preallocated with src's shape) */
void dav2_copy_relu(dav2_tensor_t *dst, const dav2_tensor_t *src);
void dav2_gelu(dav2_tensor_t *t);

/* NHWC bilinear resize with align_corners=true, matching F.interpolate. */
void dav2_interpolate(const dav2_tensor_t *in, int h, int w,
                      int oh, int ow, dav2_tensor_t *out);

/* A producer of the input of the next GEMM or convolution (the DPT head's
 * interpolations): made row by row while the accelerator already works on
 * the first tiles. The driver calls dav2_producer_need before a tile that
 * reads input pixels up to npix, and dav2_producer_idle while it waits for
 * a job; qgemm_impl completes it before anything else reads the input. */
typedef struct dav2_producer {
    int (*step)(struct dav2_producer *p);   /* make the next part; 0 when all is made */
    int done_pix;                           /* input pixels (tensor rows) made so far */
    int total_pix;
} dav2_producer_t;
void dav2_producer_set(dav2_producer_t *p);     /* NULL: none */
void dav2_producer_need(int npix);
int  dav2_producer_idle(void);                  /* one step, if any is left: 1 */
void dav2_producer_complete(void);
extern uint64_t dav2_producer_cycles;           /* CPU cycles spent in steps */

/* Background work: a producer whose steps run in the accelerator waits
 * when the foreground producer has none. dav2_background_complete finishes
 * it (before its input changes or its output is read) and clears it. */
void dav2_background_set(dav2_producer_t *p);
void dav2_background_complete(void);

/* The plain LayerNorm (no gamma, beta) in parts: begin makes the row
 * statistics and the output scale (rp in the arena), each step one output
 * row into out_v; rows below skip are not written (out row n - skip is
 * input row n). */
typedef struct {
    dav2_producer_t base;
    dav2_tensor_t in_t;                         /* the input's header, as at begin */
    const dav2_tensor_t *in;
    int16_t *out_v;
    int skip, N, C, wide, n, bg;
    int32_t *rp;
    dav2_xf_t inv16, out_scale;
    int32_t omax;
} dav2_lnplain_t;
int dav2_lnplain_begin(dav2_lnplain_t *s, const dav2_tensor_t *in, int16_t *out_v, int skip);

/* dav2_add as a producer: begin finds the output scale (out's n, c and
 * scale are set at once; amax_q by the last step), each step adds
 * step_pix pixels. */
typedef struct {
    dav2_producer_t base;
    const dav2_tensor_t *a, *b;
    dav2_tensor_t *out;
    int total, step_elems, e, wide, small;
    int32_t ma, mb, omax;
    int sa, sb;
    dav2_xf_t out_scale;
} dav2_add_t;
void dav2_add_begin(dav2_add_t *st, const dav2_tensor_t *a, const dav2_tensor_t *b,
                    dav2_tensor_t *out, int step_pix);

/* dav2_interpolate as a producer: begin (index arrays and row buffers in
 * the arena; out's n, c and scale are set at once), then its steps. */
typedef struct {
    dav2_producer_t base;
    const dav2_tensor_t *in;
    dav2_tensor_t *out;
    int h, w, oh, ow, C, wide;
    int *y0a, *y1a, *wya, *x0a, *x1a, *wxa;
    int16_t *hbuf[2];
    int hrow[2];
    size_t rowlen;
    int i;                                      /* next output row */
} dav2_interp_t;
int dav2_interp_begin(dav2_interp_t *s, const dav2_tensor_t *in, int h, int w,
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
/* dav2_conv2d_ex into out, whose data (oh*ow x m int16) the caller has
 * allocated: no copy of the result. in_relu: the convolution reads relu(in)
 * (on the accelerator while it gathers, CTRL.grelu). */
void dav2_conv2d_into(const dav2_tensor_t *in, int h, int w,
                      const dav2_qw_t *wt, int k, int stride, int pad,
                      const dav2_tensor_t *res, int relu, int in_relu,
                      dav2_tensor_t *out);

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

/* Runs the full network. depth_q receives grid*14 x grid*14 int16 values
 * and *depth_scale their scale: depth = depth_q * scale, in the same units
 * as the PyTorch model's output. The caller converts to float if it needs
 * to (the PC does; the board does not). */
void dav2_infer(const dav2_cfg_t *cfg, int16_t *depth_q, dav2_xf_t *depth_scale);

/* Input image.
 *
 * The picture used to travel inside the weight blob as two tensors, which
 * meant a new image cost a full 25 MB weight transfer. It is now supplied
 * separately: the caller points the engine at a quantised int16 HWC buffer
 * (size x size x 3, values in -8191..8191) and its scale, and can do so again
 * before every dav2_infer without touching the weights. On the board the
 * buffer lives at a fixed DDR3 address the host writes over JTAG; the host
 * build passes a malloc'd array. The scale is the float32 of the .dav2img
 * file, passed as its bit pattern; it is converted exactly with integer bit
 * operations (xf_from_f32_bits). */
void dav2_set_image(const int16_t *hwc, uint32_t scale_f32_bits);

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

/* A finer split of some buckets above, printed after the table. Each
 * detail is part of one bucket; the rest of the bucket is not itemised. */
enum {
    DAV2_SUB_LN_STATS,      /* layernorm: row statistics                   */
    DAV2_SUB_LN_Y,          /* layernorm: gamma, beta, int32 intermediate  */
    DAV2_SUB_LN_OUT,        /* layernorm: requantise the intermediate      */
    DAV2_SUB_RQ_RANGE,      /* requantise: output range from row extremes  */
    DAV2_SUB_RQ_PAR,        /* requantise: per-row multiplier and bias     */
    DAV2_SUB_ATT_PREP_QK,   /* attention: gather, shift and split q, k     */
    DAV2_SUB_ATT_PREP_V,    /* attention: gather and split v^T             */
    DAV2_SUB_ATT_SOFTMAX,   /* attention: scores to probabilities          */
    DAV2_SUB_ATT_NORM,      /* attention: context / sum                    */
    DAV2_SUB_ATT_WAIT,      /* attention: waiting for the accelerator      */
    DAV2_SUB_GELU_TABLE,    /* gelu: build the table                       */
    DAV2_SUB_N
};
extern const char *const dav2_sub_name[DAV2_SUB_N];
void     dav2_sub_add(int d, uint64_t cycles);
uint64_t dav2_sub_get(int d);

/* Soft-float call counter, dav2_floatprof.c. Does nothing unless the
 * program is built with build_flags.txt (see that file). */
void     dav2_floatprof_reset(void);
void     dav2_floatprof_report(uint64_t frame_cycles);

#endif /* DAV2_H */
