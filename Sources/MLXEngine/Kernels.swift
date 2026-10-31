// Metal compute kernels for transformer inference on MLX-format weights.
// Compiled at runtime from source so the package needs no metallib build step.

enum KernelSource {
    static let source = #"""
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Parameter blocks (must match the Swift structs in GPU.swift)
// ---------------------------------------------------------------------------
struct MMParams   { int K; int O; int T; int gs; int xStride; int yStride; int o1; int o2; int hasBias; int pad; };
struct NormParams { int dim; int headsPerRow; int rowStride; float eps; };
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

// ---------------------------------------------------------------------------
// Decode path: quantized matrix-vector product (T == 1).
// Each lane consumes 16 weights per row per step (one contiguous 2*BITS-byte
// load), x is loaded once per step and reused across QROWS rows.
// ---------------------------------------------------------------------------
template <int BITS>
inline void unpack16(device const uchar* p, thread float4* q) {
    if (BITS == 4) {
        const uint2 w = *(device const uint2*)p;
        q[0] = float4(float(w.x & 0xF), float((w.x >> 4) & 0xF), float((w.x >> 8) & 0xF), float((w.x >> 12) & 0xF));
        q[1] = float4(float((w.x >> 16) & 0xF), float((w.x >> 20) & 0xF), float((w.x >> 24) & 0xF), float(w.x >> 28));
        q[2] = float4(float(w.y & 0xF), float((w.y >> 4) & 0xF), float((w.y >> 8) & 0xF), float((w.y >> 12) & 0xF));
        q[3] = float4(float((w.y >> 16) & 0xF), float((w.y >> 20) & 0xF), float((w.y >> 24) & 0xF), float(w.y >> 28));
    } else if (BITS == 8) {
        const uchar4 a = *(device const uchar4*)p, b = *(device const uchar4*)(p + 4);
        const uchar4 c = *(device const uchar4*)(p + 8), d = *(device const uchar4*)(p + 12);
        q[0] = float4(a); q[1] = float4(b); q[2] = float4(c); q[3] = float4(d);
    } else {
        unpack8<BITS>(p, q[0], q[1]);
        unpack8<BITS>(p + BITS, q[2], q[3]);
    }
}

// Fused decode matvec over up to three matrices that share K and quantization
// (q/k/v or gate/up). Output rows are concatenated: [A rows | B rows | C rows].
// p.o1, p.o2 = row counts of A and B (multiples of 8); optional per-row bias.
template <int BITS, typename T>
kernel void qmv(device const uchar* wA [[buffer(0)]],
                device const T*     sA [[buffer(1)]],
                device const T*     bA [[buffer(2)]],
                device const float* x  [[buffer(3)]],
                device float*       y  [[buffer(4)]],
                constant MMParams&  p  [[buffer(5)]],
                device const uchar* wB [[buffer(6)]],
                device const T*     sB [[buffer(7)]],
                device const T*     bB [[buffer(8)]],
                device const uchar* wC [[buffer(9)]],
                device const T*     sC [[buffer(10)]],
                device const T*     bC [[buffer(11)]],
                device const T*     biasA [[buffer(12)]],
                device const T*     biasB [[buffer(13)]],
                device const T*     biasC [[buffer(14)]],
                uint2 tgid [[threadgroup_position_in_grid]],
                uint  sgid [[simdgroup_index_in_threadgroup]],
                uint  lane [[thread_index_in_simdgroup]]) {
    const int gRow = (int(tgid.x) * QNSG + int(sgid)) * QROWS;
    if (gRow >= p.O) return;
    device const uchar* w; device const T* sc; device const T* bi; device const T* rb;
    int rowBase;
    if (gRow < p.o1) { w = wA; sc = sA; bi = bA; rb = biasA; rowBase = gRow; }
    else if (gRow < p.o1 + p.o2) { w = wB; sc = sB; bi = bB; rb = biasB; rowBase = gRow - p.o1; }
    else { w = wC; sc = sC; bi = bC; rb = biasC; rowBase = gRow - p.o1 - p.o2; }
    const int nr = min(QROWS, p.O - gRow);
    const ulong rowBytes = ulong(p.K) * BITS / 8;
    const int gpr = p.K / p.gs;
    float acc[QROWS] = {0, 0, 0, 0};
    device const uchar* wb = w + ulong(rowBase) * rowBytes;
    device const T* sb = sc + rowBase * gpr;
    device const T* bb = bi + rowBase * gpr;
    for (int k = int(lane) * 16; k < p.K; k += 512) {
        device const float4* xp = (device const float4*)(x + k);
        const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        const float xs = dot(x0 + x1 + x2 + x3, float4(1.0f));
        const int g = k / p.gs;
        const ulong off = ulong(k) * BITS / 8;
        for (int r = 0; r < nr; r++) {
            float4 q[4];
            unpack16<BITS>(wb + ulong(r) * rowBytes + off, q);
            const float d = dot(q[0], x0) + dot(q[1], x1) + dot(q[2], x2) + dot(q[3], x3);
            acc[r] += float(sb[r * gpr + g]) * d + float(bb[r * gpr + g]) * xs;
        }
    }
    for (int r = 0; r < nr; r++) {
        float v = simd_sum(acc[r]);
        if (lane == 0) {
            if (p.hasBias) v += float(rb[rowBase + r]);
            y[gRow + r] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill path: tiled GEMM on simdgroup matrices. Y[T, O] = X[T, K] · W[O, K]ᵀ
// Threadgroup tile: 32 tokens × 32 output rows, K-step 32, 4 simdgroups (2×2),
// each owning a 16×16 block of 8×8 accumulators. Weights are dequantized into
// threadgroup memory once per tile and shared by all 32 tokens.
// ---------------------------------------------------------------------------
#define GBM 32
#define GBN 32
#define GBK 32

inline void gemm_tile_mma(threadgroup const float* Xs, threadgroup const float* Ws,
                          thread simdgroup_matrix<float, 8, 8> (&C)[2][2], uint sm, uint sn) {
    for (int kk = 0; kk < GBK; kk += 8) {
        simdgroup_matrix<float, 8, 8> A[2], B[2];
        for (int i = 0; i < 2; i++)
            simdgroup_load(A[i], Xs + (sm * 16 + i * 8) * GBK + kk, GBK);
        for (int j = 0; j < 2; j++)
            simdgroup_load(B[j], Ws + (sn * 16 + j * 8) * GBK + kk, GBK, ulong2(0, 0), true);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                simdgroup_multiply_accumulate(C[i][j], A[i], B[j], C[i][j]);
    }
}

inline void gemm_load_x(threadgroup float* Xs, device const float* x, constant MMParams& p,
                        int tBase, int k0, uint tid) {
    // 32×32 floats, 128 threads → 8 floats (two float4) per thread
    for (int i = 0; i < 2; i++) {
        const int idx = int(tid) * 2 + i;          // float4 index 0..255
        const int r = idx / (GBK / 4), c4 = idx % (GBK / 4);
        float4 v = float4(0);
        if (tBase + r < p.T) v = *(device const float4*)(x + (tBase + r) * p.xStride + k0 + c4 * 4);
        *(threadgroup float4*)(Xs + r * GBK + c4 * 4) = v;
    }
}

inline void gemm_store(threadgroup float* S, thread simdgroup_matrix<float, 8, 8> (&C)[2][2],
                       device float* y, constant MMParams& p, int tBase, int oBase, uint sm, uint sn, uint tid) {
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            simdgroup_store(C[i][j], S + (sm * 16 + i * 8) * GBN + sn * 16 + j * 8, GBN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(tid); idx < GBM * GBN; idx += 128) {
        const int r = idx / GBN, c = idx % GBN;
        if (tBase + r < p.T && oBase + c < p.O) y[(tBase + r) * p.yStride + oBase + c] = S[idx];
    }
}

template <int BITS, typename T>
kernel void qgemm(device const uchar* w      [[buffer(0)]],
                  device const T*     scales [[buffer(1)]],
                  device const T*     biases [[buffer(2)]],
                  device const float* x      [[buffer(3)]],
                  device float*       y      [[buffer(4)]],
                  constant MMParams&  p      [[buffer(5)]],
                  uint2 tgid [[threadgroup_position_in_grid]],
                  uint  tid  [[thread_index_in_threadgroup]],
                  uint  sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Xs[GBM * GBK];
    threadgroup float Ws[GBN * GBK];
    const int oBase = int(tgid.x) * GBN, tBase = int(tgid.y) * GBM;
    const uint sm = sgid / 2, sn = sgid % 2;
    simdgroup_matrix<float, 8, 8> C[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) C[i][j] = simdgroup_matrix<float, 8, 8>(0);
    const ulong rowBytes = ulong(p.K) * BITS / 8;
    const int gpr = p.K / p.gs;
    // each thread dequantizes one 8-value chunk: row = tid / 4, chunk = tid % 4
    const int wr = int(tid) / 4, wc = int(tid) % 4;
    const int orow = oBase + wr;
    for (int k0 = 0; k0 < p.K; k0 += GBK) {
        gemm_load_x(Xs, x, p, tBase, k0, tid);
        float4 qa = float4(0), qb = float4(0);
        if (orow < p.O) {
            unpack8<BITS>(w + ulong(orow) * rowBytes + ulong(k0 + wc * 8) * BITS / 8, qa, qb);
            const int g = (k0 + wc * 8) / p.gs;
            const float s = float(scales[orow * gpr + g]), b = float(biases[orow * gpr + g]);
            qa = qa * s + b; qb = qb * s + b;
        }
        *(threadgroup float4*)(Ws + wr * GBK + wc * 8) = qa;
        *(threadgroup float4*)(Ws + wr * GBK + wc * 8 + 4) = qb;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_tile_mma(Xs, Ws, C, sm, sn);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    gemm_store(Xs, C, y, p, tBase, oBase, sm, sn, tid);
}

template <typename T>
kernel void dgemm(device const T*     w [[buffer(0)]],
                  device const float* x [[buffer(3)]],
                  device float*       y [[buffer(4)]],
                  constant MMParams&  p [[buffer(5)]],
                  uint2 tgid [[threadgroup_position_in_grid]],
                  uint  tid  [[thread_index_in_threadgroup]],
                  uint  sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Xs[GBM * GBK];
    threadgroup float Ws[GBN * GBK];
    const int oBase = int(tgid.x) * GBN, tBase = int(tgid.y) * GBM;
    const uint sm = sgid / 2, sn = sgid % 2;
    simdgroup_matrix<float, 8, 8> C[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) C[i][j] = simdgroup_matrix<float, 8, 8>(0);
    const int wr = int(tid) / 4, wc = int(tid) % 4;
    const int orow = oBase + wr;
    for (int k0 = 0; k0 < p.K; k0 += GBK) {
        gemm_load_x(Xs, x, p, tBase, k0, tid);
        for (int i = 0; i < 8; i++)
            Ws[wr * GBK + wc * 8 + i] = orow < p.O ? float(w[ulong(orow) * ulong(p.K) + k0 + wc * 8 + i]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_tile_mma(Xs, Ws, C, sm, sn);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    gemm_store(Xs, C, y, p, tBase, oBase, sm, sn, tid);
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

// Fused post-projection step, one threadgroup (D threads) per (head, token):
//   q heads: optional per-head RMSNorm, RoPE, written back in place
//   k heads: optional per-head RMSNorm, RoPE, appended to the K cache
//   v heads: appended to the V cache
struct QKVPostParams { int D; int nH; int nKV; int rowStride; int offset; int maxSeq; int traditional; int rotDim; float eps; int hasNorm; };

template <typename T>
kernel void qkv_post(device float*       qkv     [[buffer(0)]],
                     device const float* invFreq [[buffer(1)]],
                     device half*        kc      [[buffer(2)]],
                     device half*        vc      [[buffer(3)]],
                     device const T*     qn      [[buffer(4)]],
                     device const T*     kn      [[buffer(5)]],
                     constant QKVPostParams& p   [[buffer(6)]],
                     uint2 tg   [[threadgroup_position_in_grid]],
                     uint  tid  [[thread_index_in_threadgroup]],
                     uint  sgid [[simdgroup_index_in_threadgroup]],
                     uint  nsg  [[simdgroups_per_threadgroup]],
                     uint  lane [[thread_index_in_simdgroup]]) {
    const int h = int(tg.x), t = int(tg.y), i = int(tid);
    device float* base = qkv + t * p.rowStride + h * p.D;
    const int pos = p.offset + t;
    float v = base[i];
    if (h >= p.nH + p.nKV) {
        vc[(ulong(h - p.nH - p.nKV) * ulong(p.maxSeq) + ulong(pos)) * ulong(p.D) + i] = half(v);
        return;
    }
    const bool isQ = h < p.nH;
    threadgroup float buf[256];
    threadgroup float red[8];
    if (p.hasNorm) {
        float ss = simd_sum(v * v);
        if (lane == 0) red[sgid] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0;
        for (uint s = 0; s < nsg; s++) tot += red[s];
        v = v * rsqrt(tot / float(p.D) + p.eps) * float(isQ ? qn[i] : kn[i]);
    }
    buf[i] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float out = v;
    if (i < p.rotDim) {
        const int half_ = p.rotDim / 2;
        int fi, partner; bool first;
        if (p.traditional) { fi = i / 2; partner = i ^ 1; first = (i & 1) == 0; }
        else { fi = i % half_; first = i < half_; partner = first ? i + half_ : i - half_; }
        const float ang = float(pos) * invFreq[fi];
        const float c = precise::cos(ang), s = precise::sin(ang);
        out = first ? v * c - buf[partner] * s : v * c + buf[partner] * s;
    }
    if (isQ) base[i] = out;
    else kc[(ulong(h - p.nH) * ulong(p.maxSeq) + ulong(pos)) * ulong(p.D) + i] = half(out);
}

// h += r; y = RMSNorm(h) * w   (residual add fused with the next norm)
template <typename T>
kernel void add_rms_norm(device float*       hbuf [[buffer(0)]],
                         device const float* r    [[buffer(1)]],
                         device float*       y    [[buffer(2)]],
                         device const T*     w    [[buffer(3)]],
                         constant NormParams& p   [[buffer(4)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint tsz  [[threads_per_threadgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint nsg  [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    device float* hr = hbuf + row * p.rowStride;
    device const float* rr = r + row * p.rowStride;
    device float* yr = y + row * p.rowStride;
    float ss = 0.0f;
    for (int i = int(tid); i < p.dim; i += int(tsz)) { const float v = hr[i] + rr[i]; hr[i] = v; ss += v * v; }
    ss = simd_sum(ss);
    threadgroup float buf[32];
    if (lane == 0) buf[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint s = 0; s < nsg; s++) tot += buf[s];
    const float inv = rsqrt(tot / float(p.dim) + p.eps);
    for (int i = int(tid); i < p.dim; i += int(tsz)) yr[i] = hr[i] * inv * float(w[i]);
}

// Causal attention over the KV cache with grouped-query heads.
// One threadgroup per (head, query token); simdgroups split the keys and
// keep an online softmax, then merge their partial results.
template <int E, int ANSG>
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
#define INST_QNEW(B, T, TN) \
  template [[host_name("qmv_" #B "_" #TN)]] kernel void qmv<B, T>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, uint2, uint, uint); \
  template [[host_name("qgemm_" #B "_" #TN)]] kernel void qgemm<B, T>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint);
#define INST_QMM_ALL(B) \
  INST_QNEW(B, half, f16) INST_QNEW(B, bfloat, bf16) INST_QNEW(B, float, f32) \
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
  template [[host_name("dgemm_" #TN)]] kernel void dgemm<T>(device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint); \
  template [[host_name("dembed_" #TN)]] kernel void dembed<T>(device const T*, device const int*, device float*, constant MMParams&, uint2); \
  template [[host_name("rms_norm_" #TN)]] kernel void rms_norm<T>(device const float*, device float*, device const T*, constant NormParams&, uint, uint, uint, uint, uint, uint); \
  template [[host_name("qkv_post_" #TN)]] kernel void qkv_post<T>(device float*, device const float*, device half*, device half*, device const T*, device const T*, constant QKVPostParams&, uint2, uint, uint, uint, uint); \
  template [[host_name("add_rms_norm_" #TN)]] kernel void add_rms_norm<T>(device float*, device const float*, device float*, device const T*, constant NormParams&, uint, uint, uint, uint, uint, uint); \
  template [[host_name("add_bias_" #TN)]] kernel void add_bias<T>(device float*, device const T*, constant EWParams&, uint2);

INST_DENSE(half, f16)
INST_DENSE(bfloat, bf16)
INST_DENSE(float, f32)

#define INST_ATTN(E) \
  template [[host_name("attention_" #E "_8")]] kernel void attention<E, 8>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint2, uint, uint); \
  template [[host_name("attention_" #E "_16")]] kernel void attention<E, 16>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint2, uint, uint); \
  template [[host_name("attention_" #E "_32")]] kernel void attention<E, 32>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint2, uint, uint);
INST_ATTN(1)
INST_ATTN(2)
INST_ATTN(3)
INST_ATTN(4)
INST_ATTN(5)
INST_ATTN(6)
INST_ATTN(8)
"""#
}
