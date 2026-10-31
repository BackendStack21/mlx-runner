// Metal compute kernels for transformer inference on MLX-format weights.
// Compiled at runtime from source so the package needs no metallib build step.

enum KernelSource {
    static let source = #"""
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Parameter blocks (must match the Swift structs in GPU.swift)
// ---------------------------------------------------------------------------
struct MMParams   { int K; int O; int T; int gs; int xStride; int yStride; };
struct NormParams { int dim; int headsPerRow; int rowStride; float eps; };
struct RopeParams { int D; int rowStride; int offset; int traditional; int rotDim; };
struct KVParams   { int D; int nKV; int rowStride; int kCol; int vCol; int offset; int maxSeq; };
struct AttnParams { int D; int nH; int nKV; int qStride; int maxSeq; int offset; float scale; int outStride; };
struct EWParams   { int n; int stride; int cols; int pad; };

#define QROWS 4
#define QNSG 2

// ---------------------------------------------------------------------------
// MLX affine quantization: 8 values are packed little-endian into BITS bytes.
// w = scale * q + bias, one (scale, bias) per group of `gs` input values.
// ---------------------------------------------------------------------------
template <int BITS>
inline void unpack8(device const uchar* p, thread float4& a, thread float4& b) {
    if (BITS == 4) {
        const uint w = *(device const uint*)p;
        a = float4(float(w & 0xF), float((w >> 4) & 0xF), float((w >> 8) & 0xF), float((w >> 12) & 0xF));
        b = float4(float((w >> 16) & 0xF), float((w >> 20) & 0xF), float((w >> 24) & 0xF), float(w >> 28));
    } else if (BITS == 8) {
        const uchar4 lo = *(device const uchar4*)p;
        const uchar4 hi = *(device const uchar4*)(p + 4);
        a = float4(lo); b = float4(hi);
    } else if (BITS == 2) {
        const uint w = uint(*(device const ushort*)p);
        a = float4(float(w & 3), float((w >> 2) & 3), float((w >> 4) & 3), float((w >> 6) & 3));
        b = float4(float((w >> 8) & 3), float((w >> 10) & 3), float((w >> 12) & 3), float((w >> 14) & 3));
    } else {
        ulong v = 0;
        for (int i = 0; i < BITS; i++) v |= ulong(p[i]) << (8 * i);
        const ulong m = (1ul << BITS) - 1ul;
        a = float4(float(v & m), float((v >> BITS) & m), float((v >> (2*BITS)) & m), float((v >> (3*BITS)) & m));
        b = float4(float((v >> (4*BITS)) & m), float((v >> (5*BITS)) & m), float((v >> (6*BITS)) & m), float((v >> (7*BITS)) & m));
    }
}

// Quantized matmul: y[t, o] = sum_k x[t, k] * W[o, k].
// Each simdgroup computes QROWS output rows for TT tokens, so every weight byte
// is read once per TT tokens (TT = 1 for decode, larger for prefill).
template <int BITS, typename T, int TT>
kernel void qmm(device const uchar* w      [[buffer(0)]],
                device const T*     scales [[buffer(1)]],
                device const T*     biases [[buffer(2)]],
                device const float* x      [[buffer(3)]],
                device float*       y      [[buffer(4)]],
                constant MMParams&  p      [[buffer(5)]],
                uint2 tgid [[threadgroup_position_in_grid]],
                uint  sgid [[simdgroup_index_in_threadgroup]],
                uint  lane [[thread_index_in_simdgroup]]) {
    const int rowBase = (int(tgid.x) * QNSG + int(sgid)) * QROWS;
    const int tBase = int(tgid.y) * TT;
    if (rowBase >= p.O) return;
    const int nt = min(TT, p.T - tBase);
    const int chunks = p.K / 8;
    const ulong rowBytes = ulong(p.K) * BITS / 8;
    const int gpr = p.K / p.gs;
    const int nr = min(QROWS, p.O - rowBase);

    float acc[QROWS][TT];
    for (int r = 0; r < QROWS; r++) for (int t = 0; t < TT; t++) acc[r][t] = 0.0f;

    device const uchar* wbase = w + ulong(rowBase) * rowBytes;
    device const T* sbase = scales + rowBase * gpr;
    device const T* bbase = biases + rowBase * gpr;

    for (int c = int(lane); c < chunks; c += 32) {
        const int k = c * 8;
        const int g = k / p.gs;
        float4 xa[TT], xb[TT];
        float xs[TT];
        for (int t = 0; t < TT; t++) {
            if (t < nt) {
                device const float4* xp = (device const float4*)(x + (tBase + t) * p.xStride + k);
                xa[t] = xp[0]; xb[t] = xp[1];
                xs[t] = dot(xa[t] + xb[t], float4(1.0f));
            } else { xa[t] = float4(0); xb[t] = float4(0); xs[t] = 0; }
        }
        for (int r = 0; r < nr; r++) {
            float4 qa, qb;
            unpack8<BITS>(wbase + ulong(r) * rowBytes + ulong(c) * BITS, qa, qb);
            const float s = float(sbase[r * gpr + g]);
            const float b = float(bbase[r * gpr + g]);
            for (int t = 0; t < TT; t++)
                acc[r][t] += s * (dot(qa, xa[t]) + dot(qb, xb[t])) + b * xs[t];
        }
    }
    for (int r = 0; r < nr; r++) {
        for (int t = 0; t < TT; t++) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0 && t < nt) y[(tBase + t) * p.yStride + rowBase + r] = v;
        }
    }
}

// Dense matmul with half / bfloat / float weights.
template <typename T, int TT>
kernel void dmm(device const T*     w [[buffer(0)]],
                device const float* x [[buffer(3)]],
                device float*       y [[buffer(4)]],
                constant MMParams&  p [[buffer(5)]],
                uint2 tgid [[threadgroup_position_in_grid]],
                uint  sgid [[simdgroup_index_in_threadgroup]],
                uint  lane [[thread_index_in_simdgroup]]) {
    const int rowBase = (int(tgid.x) * QNSG + int(sgid)) * QROWS;
    const int tBase = int(tgid.y) * TT;
    if (rowBase >= p.O) return;
    const int nt = min(TT, p.T - tBase);
    const int chunks = p.K / 4;
    const int nr = min(QROWS, p.O - rowBase);
    float acc[QROWS][TT];
    for (int r = 0; r < QROWS; r++) for (int t = 0; t < TT; t++) acc[r][t] = 0.0f;
    for (int c = int(lane); c < chunks; c += 32) {
        const int k = c * 4;
        float4 xv[TT];
        for (int t = 0; t < TT; t++)
            xv[t] = (t < nt) ? *(device const float4*)(x + (tBase + t) * p.xStride + k) : float4(0);
        for (int r = 0; r < nr; r++) {
            device const T* wp = w + ulong(rowBase + r) * ulong(p.K) + k;
            const float4 wv = float4(float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3]));
            for (int t = 0; t < TT; t++) acc[r][t] += dot(wv, xv[t]);
        }
    }
    for (int r = 0; r < nr; r++) {
        for (int t = 0; t < TT; t++) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0 && t < nt) y[(tBase + t) * p.yStride + rowBase + r] = v;
        }
    }
}

// Embedding lookups -------------------------------------------------------
template <int BITS, typename T>
kernel void qembed(device const uchar* w      [[buffer(0)]],
                   device const T*     scales [[buffer(1)]],
                   device const T*     biases [[buffer(2)]],
                   device const int*   ids    [[buffer(3)]],
                   device float*       out    [[buffer(4)]],
                   constant MMParams&  p      [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
    const int c = int(gid.x), t = int(gid.y);
    if (c >= p.K / 8 || t >= p.T) return;
    const int row = ids[t];
    const ulong rowBytes = ulong(p.K) * BITS / 8;
    float4 qa, qb;
    unpack8<BITS>(w + ulong(row) * rowBytes + ulong(c) * BITS, qa, qb);
    const int gpr = p.K / p.gs;
    const int g = (c * 8) / p.gs;
    const float s = float(scales[row * gpr + g]);
    const float b = float(biases[row * gpr + g]);
    device float4* o = (device float4*)(out + t * p.K + c * 8);
    o[0] = qa * s + b;
    o[1] = qb * s + b;
}

template <typename T>
kernel void dembed(device const T*    w   [[buffer(0)]],
                   device const int*  ids [[buffer(3)]],
                   device float*      out [[buffer(4)]],
                   constant MMParams& p   [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
    const int k = int(gid.x), t = int(gid.y);
    if (k >= p.K || t >= p.T) return;
    out[t * p.K + k] = float(w[ulong(ids[t]) * ulong(p.K) + k]);
}

// RMSNorm over rows; also used per-head for q_norm / k_norm. ----------------
template <typename T>
kernel void rms_norm(device const float* x [[buffer(0)]],
                     device float*       y [[buffer(1)]],
                     device const T*     w [[buffer(2)]],
                     constant NormParams& p [[buffer(3)]],
                     uint row  [[threadgroup_position_in_grid]],
                     uint tid  [[thread_index_in_threadgroup]],
                     uint tsz  [[threads_per_threadgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint nsg  [[simdgroups_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const int t = int(row) / p.headsPerRow, h = int(row) % p.headsPerRow;
    device const float* xr = x + t * p.rowStride + h * p.dim;
    device float* yr = y + t * p.rowStride + h * p.dim;
    float ss = 0.0f;
    for (int i = int(tid); i < p.dim; i += int(tsz)) { const float v = xr[i]; ss += v * v; }
    ss = simd_sum(ss);
    threadgroup float buf[32];
    if (lane == 0) buf[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = lane < nsg ? buf[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(buf[0] / float(p.dim) + p.eps);
    for (int i = int(tid); i < p.dim; i += int(tsz)) yr[i] = xr[i] * inv * float(w[i]);
}

// Rotary position embedding, applied in place to the q and k heads of the fused qkv buffer.
kernel void rope(device float*       qkv     [[buffer(0)]],
                 device const float* invFreq [[buffer(1)]],
                 constant RopeParams& p      [[buffer(2)]],
                 uint3 gid [[thread_position_in_grid]]) {
    const int i = int(gid.x), h = int(gid.y), t = int(gid.z);
    const int half_ = p.rotDim / 2;
    if (i >= half_) return;
    const float ang = float(p.offset + t) * invFreq[i];
    const float c = precise::cos(ang), s = precise::sin(ang);
    device float* base = qkv + t * p.rowStride + h * p.D;
    const int i0 = p.traditional ? 2 * i : i;
    const int i1 = p.traditional ? 2 * i + 1 : i + half_;
    const float x0 = base[i0], x1 = base[i1];
    base[i0] = x0 * c - x1 * s;
    base[i1] = x1 * c + x0 * s;
}

// Append K/V for the current tokens into the half-precision cache.
kernel void kv_store(device const float* qkv [[buffer(0)]],
                     device half*        kc  [[buffer(1)]],
                     device half*        vc  [[buffer(2)]],
                     constant KVParams&  p   [[buffer(3)]],
                     uint3 gid [[thread_position_in_grid]]) {
    const int d = int(gid.x), h = int(gid.y), t = int(gid.z);
    if (d >= p.D) return;
    device const float* src = qkv + t * p.rowStride;
    const ulong dst = (ulong(h) * ulong(p.maxSeq) + ulong(p.offset + t)) * ulong(p.D) + d;
    kc[dst] = half(src[p.kCol + h * p.D + d]);
    vc[dst] = half(src[p.vCol + h * p.D + d]);
}

// Causal attention over the KV cache with grouped-query heads.
// One threadgroup per (head, query token); simdgroups split the keys and
// keep an online softmax, then merge their partial results.
#define ANSG 8
template <int E>
kernel void attention(device const float* qkv [[buffer(0)]],
                      device const half*  kc  [[buffer(1)]],
                      device const half*  vc  [[buffer(2)]],
                      device float*       out [[buffer(3)]],
                      constant AttnParams& p  [[buffer(4)]],
                      uint2 tg   [[threadgroup_position_in_grid]],
                      uint  sg   [[simdgroup_index_in_threadgroup]],
                      uint  lane [[thread_index_in_simdgroup]]) {
    const int h = int(tg.x), t = int(tg.y);
    const int kvh = h / (p.nH / p.nKV);
    device const float* q = qkv + t * p.qStride + h * p.D;
    float qr[E], acc[E];
    for (int e = 0; e < E; e++) { qr[e] = q[e * 32 + lane] * p.scale; acc[e] = 0.0f; }
    float m = -1e30f, l = 0.0f;
    const int L = p.offset + t + 1;
    device const half* kb = kc + ulong(kvh) * ulong(p.maxSeq) * ulong(p.D);
    device const half* vb = vc + ulong(kvh) * ulong(p.maxSeq) * ulong(p.D);
    for (int j = int(sg); j < L; j += ANSG) {
        device const half* kp = kb + ulong(j) * ulong(p.D);
        float s = 0.0f;
        for (int e = 0; e < E; e++) s += qr[e] * float(kp[e * 32 + lane]);
        s = simd_sum(s);
        const float mn = max(m, s);
        const float cr = exp(m - mn), pe = exp(s - mn);
        l = l * cr + pe;
        device const half* vp = vb + ulong(j) * ulong(p.D);
        for (int e = 0; e < E; e++) acc[e] = acc[e] * cr + pe * float(vp[e * 32 + lane]);
        m = mn;
    }
    threadgroup float tm[ANSG], tl[ANSG];
    threadgroup float ta[ANSG * E * 32];
    if (lane == 0) { tm[sg] = m; tl[sg] = l; }
    for (int e = 0; e < E; e++) ta[sg * E * 32 + e * 32 + lane] = acc[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float M = -1e30f;
        for (int s = 0; s < ANSG; s++) M = max(M, tm[s]);
        float Ls = 0.0f;
        float o[E];
        for (int e = 0; e < E; e++) o[e] = 0.0f;
        for (int s = 0; s < ANSG; s++) {
            const float f = exp(tm[s] - M);
            Ls += tl[s] * f;
            for (int e = 0; e < E; e++) o[e] += ta[s * E * 32 + e * 32 + lane] * f;
        }
        device float* op = out + t * p.outStride + h * p.D;
        const float inv = 1.0f / Ls;
        for (int e = 0; e < E; e++) op[e * 32 + lane] = o[e] * inv;
    }
}

// Elementwise ---------------------------------------------------------------
// out[t, i] = silu(gu[t, i]) * gu[t, cols + i]
kernel void silu_mul(device const float* gu  [[buffer(0)]],
                     device float*       out [[buffer(1)]],
                     constant EWParams&  p   [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const int i = int(gid.x), t = int(gid.y);
    if (i >= p.cols) return;
    const float g = gu[t * p.stride + i];
    const float u = gu[t * p.stride + p.cols + i];
    out[t * p.cols + i] = (g / (1.0f + exp(-g))) * u;
}

kernel void add_inplace(device float*       a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        constant EWParams&  p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (int(gid) < p.n) a[gid] += b[gid];
}

template <typename T>
kernel void add_bias(device float*      y [[buffer(0)]],
                     device const T*    b [[buffer(1)]],
                     constant EWParams& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const int o = int(gid.x), t = int(gid.y);
    if (o >= p.cols) return;
    y[t * p.stride + o] += float(b[o]);
}

// Explicit instantiations ----------------------------------------------------
#define INST_QMM(B, T, TN, TT) \
  template [[host_name("qmm_" #B "_" #TN "_" #TT)]] kernel void qmm<B, T, TT>( \
    device const uchar*, device const T*, device const T*, device const float*, device float*, \
    constant MMParams&, uint2, uint, uint);
#define INST_QMM_ALL(B) \
  INST_QMM(B, half, f16, 1) INST_QMM(B, half, f16, 8) \
  INST_QMM(B, bfloat, bf16, 1) INST_QMM(B, bfloat, bf16, 8) \
  INST_QMM(B, float, f32, 1) INST_QMM(B, float, f32, 8) \
  template [[host_name("qembed_" #B "_f16")]] kernel void qembed<B, half>(device const uchar*, device const half*, device const half*, device const int*, device float*, constant MMParams&, uint2); \
  template [[host_name("qembed_" #B "_bf16")]] kernel void qembed<B, bfloat>(device const uchar*, device const bfloat*, device const bfloat*, device const int*, device float*, constant MMParams&, uint2); \
  template [[host_name("qembed_" #B "_f32")]] kernel void qembed<B, float>(device const uchar*, device const float*, device const float*, device const int*, device float*, constant MMParams&, uint2);

INST_QMM_ALL(2)
INST_QMM_ALL(3)
INST_QMM_ALL(4)
INST_QMM_ALL(5)
INST_QMM_ALL(6)
INST_QMM_ALL(8)

#define INST_DENSE(T, TN) \
  template [[host_name("dmm_" #TN "_1")]] kernel void dmm<T, 1>(device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint); \
  template [[host_name("dmm_" #TN "_8")]] kernel void dmm<T, 8>(device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint); \
  template [[host_name("dembed_" #TN)]] kernel void dembed<T>(device const T*, device const int*, device float*, constant MMParams&, uint2); \
  template [[host_name("rms_norm_" #TN)]] kernel void rms_norm<T>(device const float*, device float*, device const T*, constant NormParams&, uint, uint, uint, uint, uint, uint); \
  template [[host_name("add_bias_" #TN)]] kernel void add_bias<T>(device float*, device const T*, constant EWParams&, uint2);

INST_DENSE(half, f16)
INST_DENSE(bfloat, bf16)
INST_DENSE(float, f32)

#define INST_ATTN(E) \
  template [[host_name("attention_" #E)]] kernel void attention<E>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint2, uint, uint);
INST_ATTN(1)
INST_ATTN(2)
INST_ATTN(3)
INST_ATTN(4)
INST_ATTN(5)
INST_ATTN(6)
INST_ATTN(8)
"""#
}
