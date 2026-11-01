// Metal compute kernels for transformer inference on MLX-format weights.
// Compiled at runtime from source so the package needs no metallib build step.

enum KernelSource {
    static let source = #"""
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Parameter blocks (must match the Swift structs in GPU.swift)
// ---------------------------------------------------------------------------
struct MMParams   { int K; int O; int T; int gs; int xStride; int yStride; int o1; int o2; int hasBias; int act; };
struct NormParams { int dim; int headsPerRow; int rowStride; float eps; };
struct AttnParams { int D; int nH; int nKV; int qStride; int maxSeq; int offset; float scale; int outStride; int T; int window; float softcap; };
// Sliding window (window > 0): the query at position i sees keys i - window < j <= i.
inline int attn_first_key(constant AttnParams& p, int pos) { return p.window > 0 ? max(0, pos - p.window + 1) : 0; }
// K/V cache row of position j: caches hold `len` rows (AttnParams / QKVPostParams.maxSeq) and
// a sliding-window layer keeps only the last ones, position j living at row j mod len.
inline int kv_slot(int j, int len) { return j < len ? j : j % len; }
// Gemma 2 attention-score soft-capping: s = tanh(s / cap) * cap. A template parameter of the
// attention kernels (`_cap` instantiations) so the other models' inner loops do not carry it.
template <bool CAP>
inline float attn_cap(float s, float cap) { return CAP ? precise::tanh(s / cap) * cap : s; }
struct EWParams   { int n; int stride; int cols; int pad; };

#define QROWS 4
#define QNSG 2
#define QVNSG 2

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

// erf with |error| < 1.2e-7 (Numerical Recipes erfc, Chebyshev fit); Metal has no erf.
inline float erf_nr(float x) {
    const float z = fabs(x), t = 1.0f / (1.0f + 0.5f * z);
    const float r = t * precise::exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
                    t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
                    t * (-0.82215223f + t * 0.17087277f)))))))));
    return x >= 0.0f ? 1.0f - r : r - 1.0f;
}

// Gated activations (0 SiLU, 1 tanh GELU, 2 exact GELU): act(g) * u. One definition for the
// elementwise kernels and the fused matvec epilogue, so both produce the same values.
inline float act_mul(int act, float g, float u) {
    if (act == 0) return (g / (1.0f + exp(-g))) * u;
    if (act == 1) return 0.5f * g * (1.0f + precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))) * u;
    return 0.5f * g * (1.0f + erf_nr(g * 0.7071067811865476f)) * u;
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

// Remainder pass of the 4-bit decode matvec: CW (4 or 8) values per lane starting at
// k0 + lane * CW, every lane active. CW divides the group size, so a lane stays in one group.
// Trip counts are constants: the x registers must not spill to private memory.
template <int CW, typename T>
inline void qmv4_tail(device const uchar* wb, ulong rowBytes, device const T* sb, device const T* bb, int gpr,
                      device const float* x, int k0, int nr, int gs, thread float* acc, uint lane) {
    const float4 sLo = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    const float4 sHi = sLo * (1.0f / 65536.0f);
    const uint4 mLo = uint4(0xFu, 0xF0u, 0xF00u, 0xF000u);
    const uint4 mHi = mLo << 16;
    const int k = k0 + int(lane) * CW;
    const int g = k / gs;
    const ulong off = ulong(k) >> 1;
    device const float4* xp = (device const float4*)(x + k);
    float4 xv[CW / 4];
    float xs = 0.0f;
    for (int i = 0; i < CW / 4; i++) {
        const float4 v = xp[i];
        xs += v.x + v.y + v.z + v.w;
        xv[i] = v * ((i & 1) ? sHi : sLo);
    }
    for (int r = 0; r < nr; r++) {
        device const uchar* wp = wb + ulong(r) * rowBytes + off;
        float d = 0.0f;
        if (CW == 4) {
            d = dot(float4(uint4(uint(*(device const ushort*)wp)) & mLo), xv[0]);
        } else {
            for (int i = 0; i < CW / 8; i++) {
                const uint4 wi = uint4(((device const uint*)wp)[i]);
                d += dot(float4(wi & mLo), xv[2 * i]) + dot(float4(wi & mHi), xv[(2 * i + 1) % (CW / 4)]);
            }
        }
        acc[r] += float(sb[r * gpr + g]) * d + float(bb[r * gpr + g]) * xs;
    }
}

// Fused decode matvec over up to three matrices that share K and quantization
// (q/k/v or gate/up). Output rows are concatenated: [A rows | B rows | C rows].
// p.o1, p.o2 = row counts of A and B (multiples of 8); optional per-row bias.
// MODE 1 (gated output): y[row] = act(y[row]) · e[row] (Gemma 4 per-layer input gate).
// TAIL (4-bit only; the host picks it from K % 1024): the remainder after the 1024-wide
// passes is 128 (TAIL 4: 4 values per lane), 256 (TAIL 8) or 512 (TAIL 16: 16-byte loads,
// lanes 0-15 on one row and 16-31 on the next). A separate instantiation per tail keeps the
// plain kernel's code (and occupancy) unchanged: tail code inside one kernel cost every
// model 2-11%.
// Measured slower in the full decode step and rejected: 2 rows per simdgroup for small
// matrices; a gate/up fusion computing 8 rows per simdgroup with the activation in the
// epilogue (−4% to −11% on every model); split-K with the two simdgroups sharing 4 rows
// and alternating passes (−3% to −11% on Gemma).
template <int BITS, typename T, int MODE, int TAIL>
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
                device const float* e     [[buffer(15)]],
                uint2 tgid [[threadgroup_position_in_grid]],
                uint  sgid [[simdgroup_index_in_threadgroup]],
                uint  lane [[thread_index_in_simdgroup]]) {
    const int gRow = (int(tgid.x) * QVNSG + int(sgid)) * QROWS;
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
    if (BITS == 4) {
        // Fast 4-bit path: 32 weights (16 bytes) per lane per row per step.
        // x is pre-scaled by 16^-j so each nibble is extracted with a mask only
        // (no shifts): (w & 0xF0) * x/16 == ((w >> 4) & 0xF) * x.
        const float4 sLo = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
        const float4 sHi = sLo * (1.0f / 65536.0f);
        const uint4 mLo = uint4(0xFu, 0xF0u, 0xF00u, 0xF000u);
        const uint4 mHi = mLo << 16;
        // A pass covers 1024 values; TAIL handles the remainder (see above): +10% decode on
        // Gemma 3 1B (1152) versus a pass with 4 lanes busy.
        const int full = TAIL == 0 ? p.K : (p.K & ~1023);
        for (int k = int(lane) * 32; k < full; k += 1024) {
            device const float4* xp = (device const float4*)(x + k);
            float4 xv[8];
            float xs = 0.0f;
            for (int i = 0; i < 8; i++) {
                const float4 v = xp[i];
                xs += v.x + v.y + v.z + v.w;
                xv[i] = v * ((i & 1) ? sHi : sLo);
            }
            const int g = k / p.gs;
            const ulong off = ulong(k) >> 1;
            for (int r = 0; r < nr; r++) {
                const uint4 w = *(device const uint4*)(wb + ulong(r) * rowBytes + off);
                float d = 0.0f;
                for (int i = 0; i < 4; i++) {
                    const uint4 wi = uint4(w[i]);
                    d += dot(float4(wi & mLo), xv[2 * i]) + dot(float4(wi & mHi), xv[2 * i + 1]);
                }
                acc[r] += float(sb[r * gpr + g]) * d + float(bb[r * gpr + g]) * xs;
            }
        }
        if (TAIL == 16) {
            const int k = full + int(lane % 16) * 32;
            device const float4* xp = (device const float4*)(x + k);
            float4 xv[8];
            float xs = 0.0f;
            for (int i = 0; i < 8; i++) {
                const float4 v = xp[i];
                xs += v.x + v.y + v.z + v.w;
                xv[i] = v * ((i & 1) ? sHi : sLo);
            }
            const int g = k / p.gs;
            const ulong off = ulong(k) >> 1;
            for (int r = int(lane) / 16; r < nr; r += 2) {   // the other half's lanes add nothing to this row
                const uint4 w = *(device const uint4*)(wb + ulong(r) * rowBytes + off);
                float d = 0.0f;
                for (int i = 0; i < 4; i++) {
                    const uint4 wi = uint4(w[i]);
                    d += dot(float4(wi & mLo), xv[2 * i]) + dot(float4(wi & mHi), xv[2 * i + 1]);
                }
                acc[r] += float(sb[r * gpr + g]) * d + float(bb[r * gpr + g]) * xs;
            }
        } else if (TAIL == 8) {
            qmv4_tail<8>(wb, rowBytes, sb, bb, gpr, x, full, nr, p.gs, acc, lane);
        } else if (TAIL == 4) {
            qmv4_tail<4>(wb, rowBytes, sb, bb, gpr, x, full, nr, p.gs, acc, lane);
        }
    } else {
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
    }
    for (int r = 0; r < nr; r++) {
        float v = simd_sum(acc[r]);
        if (lane == 0) {
            if (p.hasBias) v += float(rb[rowBase + r]);
            y[gRow + r] = MODE == 1 ? act_mul(p.act, v, e[gRow + r]) : v;
        }
    }
}

// Small-batch 4-bit matvec for speculative verification (2..TT tokens): each
// weight chunk is loaded once and applied to every token, so verifying a few
// draft tokens costs little more than decoding one. Same fused [A|B|C] layout.
template <typename T, int TT, int R>
kernel void qmvb4(device const uchar* wA [[buffer(0)]],
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
    const int gRow = (int(tgid.x) * QVNSG + int(sgid)) * R;
    if (gRow >= p.O) return;
    device const uchar* w; device const T* sc; device const T* bi; device const T* rb;
    int rowBase;
    if (gRow < p.o1) { w = wA; sc = sA; bi = bA; rb = biasA; rowBase = gRow; }
    else if (gRow < p.o1 + p.o2) { w = wB; sc = sB; bi = bB; rb = biasB; rowBase = gRow - p.o1; }
    else { w = wC; sc = sC; bi = bC; rb = biasC; rowBase = gRow - p.o1 - p.o2; }
    const int nr = min(R, p.O - gRow);
    const int nt = min(TT, p.T);
    const ulong rowBytes = ulong(p.K) / 2;
    const int gpr = p.K / p.gs;
    float acc[R][TT];
    for (int r = 0; r < R; r++) for (int t = 0; t < TT; t++) acc[r][t] = 0.0f;
    device const uchar* wb = w + ulong(rowBase) * rowBytes;
    device const T* sb = sc + rowBase * gpr;
    device const T* bb = bi + rowBase * gpr;
    const float4 sLo = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
    const float4 sHi = sLo * (1.0f / 65536.0f);
    const uint4 mLo = uint4(0xFu, 0xF0u, 0xF00u, 0xF000u);
    const uint4 mHi = mLo << 16;
    // 8 weights (one packed uint) per lane per step: the activations of every token
    // fit in registers, and each weight is dequantised once and reused by all tokens.
    for (int k = int(lane) * 8; k < p.K; k += 256) {
        float4 xl[TT], xh[TT];
        float xs[TT];
        for (int t = 0; t < TT; t++) {
            device const float4* xp = (device const float4*)(x + min(t, nt - 1) * p.xStride + k);
            const float4 a = xp[0], b = xp[1];
            xs[t] = dot(a + b, float4(1.0f));
            xl[t] = a * sLo; xh[t] = b * sHi;
        }
        const int g = k / p.gs;
        const ulong off = ulong(k) >> 1;
        for (int r = 0; r < R; r++) {
            const int rr = min(r, nr - 1);
            const uint wv = *(device const uint*)(wb + ulong(rr) * rowBytes + off);
            const float4 ql = float4(uint4(wv) & mLo), qh = float4(uint4(wv) & mHi);
            const float s = float(sb[rr * gpr + g]), b = float(bb[rr * gpr + g]);
            for (int t = 0; t < TT; t++)
                acc[r][t] += s * (dot(ql, xl[t]) + dot(qh, xh[t])) + b * xs[t];
        }
    }
    for (int r = 0; r < R; r++) {
        for (int t = 0; t < TT; t++) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0 && r < nr && t < nt)
                y[t * p.yStride + gRow + r] = v + (p.hasBias ? float(rb[rowBase + r]) : 0.0f);
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

// ---------------------------------------------------------------------------
// Short prompts (5..16 tokens): qgemm pads every batch to a 32-token tile, so
// 5 tokens cost as much MMA work as 32. This variant uses a BM = 8 or 16 token
// tile. Each output element sees exactly the same chain as in qgemm (fp32
// dequant s * q + b, x zero-padded, 8×8×8 multiply-accumulates with k ascending
// in steps of 8, rows and columns 8-aligned), so results match qgemm.
// 4 simdgroups each own 8 of the tile's 32 output columns. K-step 64 halves the
// barriers per MMA, and the next step's weights / activations are fetched into
// registers before the current step's MMAs run. Requires K % 64 == 0.
// ---------------------------------------------------------------------------
#define SBK 64

template <int BITS, typename T>
inline void gemm_dequant8(device const uchar* wrow, device const T* srow, device const T* brow,
                          int k, int gs, bool ok, thread float4& qa, thread float4& qb) {
    qa = float4(0); qb = float4(0);
    if (ok) {
        unpack8<BITS>(wrow + ulong(k) * BITS / 8, qa, qb);
        const int g = k / gs;
        const float s = float(srow[g]), b = float(brow[g]);
        qa = qa * s + b; qb = qb * s + b;
    }
}

template <int BITS, typename T, int BM>
kernel void qgemms(device const uchar* w      [[buffer(0)]],
                   device const T*     scales [[buffer(1)]],
                   device const T*     biases [[buffer(2)]],
                   device const float* x      [[buffer(3)]],
                   device float*       y      [[buffer(4)]],
                   constant MMParams&  p      [[buffer(5)]],
                   uint2 tgid [[threadgroup_position_in_grid]],
                   uint  tid  [[thread_index_in_threadgroup]],
                   uint  sgid [[simdgroup_index_in_threadgroup]]) {
    static_assert(BM == 8 || BM == 16, "token tile");
    constexpr int XV = BM * SBK / 4 / 128;      // float4s of x staged per thread per step (1 or 2)
    threadgroup float Xs[BM * SBK];             // also the output staging area (BM × GBN floats)
    threadgroup float Ws[GBN * SBK];
    const int oBase = int(tgid.x) * GBN, tBase = int(tgid.y) * BM;
    const int sn = int(sgid);                   // output columns sn*8 ..< sn*8+8 of the tile
    simdgroup_matrix<float, 8, 8> C[BM / 8];
    for (int i = 0; i < BM / 8; i++) C[i] = simdgroup_matrix<float, 8, 8>(0);
    const ulong rowBytes = ulong(p.K) * BITS / 8;
    const int gpr = p.K / p.gs;
    // Dequantization: thread → weight row tid / 4, 8-value chunks (tid % 4) and (tid % 4) + 4.
    const int wr = int(tid) / 4, wc = int(tid) % 4;
    const int orow = oBase + wr;
    const bool rowOK = orow < p.O;
    const ulong srcRow = rowOK ? ulong(orow) : 0ul;
    device const uchar* wrow = w + srcRow * rowBytes;
    device const T* srow = scales + srcRow * ulong(gpr);
    device const T* brow = biases + srcRow * ulong(gpr);
    // Activations: float4 index tid + 128 v → token row idx / 16, column (idx % 16) * 4.
    int xr[XV], xc[XV];
    bool xok[XV];
    for (int v = 0; v < XV; v++) {
        const int idx = int(tid) + 128 * v;
        xr[v] = idx / (SBK / 4); xc[v] = (idx % (SBK / 4)) * 4;
        xok[v] = tBase + xr[v] < p.T;
    }
    float4 wq[4], xq[XV];
    gemm_dequant8<BITS, T>(wrow, srow, brow, wc * 8, p.gs, rowOK, wq[0], wq[1]);
    gemm_dequant8<BITS, T>(wrow, srow, brow, (wc + 4) * 8, p.gs, rowOK, wq[2], wq[3]);
    for (int v = 0; v < XV; v++)
        xq[v] = xok[v] ? *(device const float4*)(x + (tBase + xr[v]) * p.xStride + xc[v]) : float4(0);
    for (int k0 = 0; k0 < p.K; k0 += SBK) {
        for (int v = 0; v < XV; v++) *(threadgroup float4*)(Xs + xr[v] * SBK + xc[v]) = xq[v];
        *(threadgroup float4*)(Ws + wr * SBK + wc * 8) = wq[0];
        *(threadgroup float4*)(Ws + wr * SBK + wc * 8 + 4) = wq[1];
        *(threadgroup float4*)(Ws + wr * SBK + (wc + 4) * 8) = wq[2];
        *(threadgroup float4*)(Ws + wr * SBK + (wc + 4) * 8 + 4) = wq[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int kn = k0 + SBK;
        if (kn < p.K) {                         // prefetch the next step while this one multiplies
            gemm_dequant8<BITS, T>(wrow, srow, brow, kn + wc * 8, p.gs, rowOK, wq[0], wq[1]);
            gemm_dequant8<BITS, T>(wrow, srow, brow, kn + (wc + 4) * 8, p.gs, rowOK, wq[2], wq[3]);
            for (int v = 0; v < XV; v++)
                xq[v] = xok[v] ? *(device const float4*)(x + (tBase + xr[v]) * p.xStride + kn + xc[v]) : float4(0);
        }
        for (int kk = 0; kk < SBK; kk += 8) {
            simdgroup_matrix<float, 8, 8> A, B;
            simdgroup_load(B, Ws + (sn * 8) * SBK + kk, SBK, ulong2(0, 0), true);
            for (int i = 0; i < BM / 8; i++) {
                simdgroup_load(A, Xs + (i * 8) * SBK + kk, SBK);
                simdgroup_multiply_accumulate(C[i], A, B, C[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < BM / 8; i++) simdgroup_store(C[i], Xs + (i * 8) * GBN + sn * 8, GBN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(tid); idx < BM * GBN; idx += 128) {
        const int r = idx / GBN, c = idx % GBN;
        if (tBase + r < p.T && oBase + c < p.O) y[(tBase + r) * p.yStride + oBase + c] = Xs[idx];
    }
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
    const int n4 = p.dim / 4;   // dim is a multiple of 4 and the rows are 16-byte aligned (checked on the host)
    float ss = 0.0f;
    for (int i = int(tid); i < n4; i += int(tsz)) { const float4 v = ((device const float4*)xr)[i]; ss += dot(v, v); }
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
    for (int i = int(tid); i < n4; i += int(tsz)) {
        const float4 v = ((device const float4*)xr)[i];
        const float4 ww = float4(float(w[4 * i]), float(w[4 * i + 1]), float(w[4 * i + 2]), float(w[4 * i + 3]));
        ((device float4*)yr)[i] = v * inv * ww;
    }
}

// Fused post-projection step, one threadgroup (D threads) per (head, token):
//   q heads: optional per-head RMSNorm, RoPE, written back in place
//   k heads: optional per-head RMSNorm, RoPE, appended to the K cache
//   v heads: optional weightless RMSNorm (Gemma 4), appended to the V cache; with kEqV they
//            read the raw k head (Gemma 4 global layers without v_proj)
// D <= 1024 (checked on the host). Dispatch only the nH q heads for a layer that reuses
// another layer's KV cache.
struct QKVPostParams { int D; int nH; int nKV; int rowStride; int offset; int maxSeq; int traditional; int rotDim; float eps; int hasNorm; int vNorm; int kEqV; };

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
    const int row = kv_slot(pos, p.maxSeq);
    const bool isV = h >= p.nH + p.nKV;
    float v = (isV && p.kEqV) ? base[i - p.nKV * p.D] : base[i];
    threadgroup float buf[1024];
    threadgroup float red[32];
    if (isV) {
        if (p.vNorm) {   // uniform per threadgroup
            float ss = simd_sum(v * v);
            if (lane == 0) red[sgid] = ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tot = 0;
            for (uint s = 0; s < nsg; s++) tot += red[s];
            v *= rsqrt(tot / float(p.D) + p.eps);
        }
        vc[(ulong(h - p.nH - p.nKV) * ulong(p.maxSeq) + ulong(row)) * ulong(p.D) + i] = half(v);
        return;
    }
    const bool isQ = h < p.nH;
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
    else kc[(ulong(h - p.nH) * ulong(p.maxSeq) + ulong(row)) * ulong(p.D) + i] = half(out);
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
    const int n4 = p.dim / 4;   // dim is a multiple of 4 and the rows are 16-byte aligned (checked on the host)
    float ss = 0.0f;
    for (int i = int(tid); i < n4; i += int(tsz)) {
        const float4 v = ((device const float4*)hr)[i] + ((device const float4*)rr)[i];
        ((device float4*)hr)[i] = v;
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    threadgroup float buf[32];
    if (lane == 0) buf[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint s = 0; s < nsg; s++) tot += buf[s];
    const float inv = rsqrt(tot / float(p.dim) + p.eps);
    for (int i = int(tid); i < n4; i += int(tsz)) {
        const float4 v = ((device const float4*)hr)[i];
        const float4 ww = float4(float(w[4 * i]), float(w[4 * i + 1]), float(w[4 * i + 2]), float(w[4 * i + 3]));
        ((device float4*)yr)[i] = v * inv * ww;
    }
}

// h = (h + RMSNorm(r) * wr) * hScale; y = RMSNorm(h) * w   (Gemma 2/3/4: the block output is
// normalized before the residual add, fused with the next norm; hScale is Gemma 4's layer_scalar)
template <typename T>
kernel void norm_add_rms_norm(device float*       hbuf [[buffer(0)]],
                              device const float* r    [[buffer(1)]],
                              device float*       y    [[buffer(2)]],
                              device const T*     w    [[buffer(3)]],
                              constant NormParams& p   [[buffer(4)]],
                              device const T*     wr   [[buffer(5)]],
                              constant float&     hScale [[buffer(6)]],
                              uint row  [[threadgroup_position_in_grid]],
                              uint tid  [[thread_index_in_threadgroup]],
                              uint tsz  [[threads_per_threadgroup]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint nsg  [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    device float* hr = hbuf + row * p.rowStride;
    device const float* rr = r + row * p.rowStride;
    device float* yr = y + row * p.rowStride;
    const int n4 = p.dim / 4;   // dim is a multiple of 4 and the rows are 16-byte aligned (checked on the host)
    threadgroup float buf[2][32];
    float ss = 0.0f;
    for (int i = int(tid); i < n4; i += int(tsz)) { const float4 v = ((device const float4*)rr)[i]; ss += dot(v, v); }
    ss = simd_sum(ss);
    if (lane == 0) buf[0][sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint s = 0; s < nsg; s++) tot += buf[0][s];
    const float invR = rsqrt(tot / float(p.dim) + p.eps);
    ss = 0.0f;
    for (int i = int(tid); i < n4; i += int(tsz)) {
        const float4 ww = float4(float(wr[4 * i]), float(wr[4 * i + 1]), float(wr[4 * i + 2]), float(wr[4 * i + 3]));
        const float4 v = (((device const float4*)hr)[i] + ((device const float4*)rr)[i] * invR * ww) * hScale;
        ((device float4*)hr)[i] = v;
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    if (lane == 0) buf[1][sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tot = 0;
    for (uint s = 0; s < nsg; s++) tot += buf[1][s];
    const float inv = rsqrt(tot / float(p.dim) + p.eps);
    for (int i = int(tid); i < n4; i += int(tsz)) {
        const float4 v = ((device const float4*)hr)[i];
        const float4 ww = float4(float(w[4 * i]), float(w[4 * i + 1]), float(w[4 * i + 2]), float(w[4 * i + 3]));
        ((device float4*)yr)[i] = v * inv * ww;
    }
}

// Causal attention over the KV cache with grouped-query heads.
// One threadgroup per (head, query token); ANSG simdgroups split the keys
// (simdgroup s takes keys s, s+ANSG, ...) and keep an online softmax, then
// merge their partial results in simdgroup order. AU keys' K/V rows can be
// loaded up front (same arithmetic order as one key at a time); measured on an
// M2 Pro, AU = 4 was no faster than 1 at 16 tokens of context, so it stays 1.
#define AU 1

// One online-softmax step for one key: k/v are this lane's E elements.
template <int E, bool CAP>
inline void attn_update(thread const float* qr, thread float* acc, thread float& m, thread float& l,
                        thread const half* k, thread const half* v, float cap) {
    float s = 0.0f;
    for (int e = 0; e < E; e++) s += qr[e] * float(k[e]);
    s = attn_cap<CAP>(simd_sum(s), cap);
    const float mn = max(m, s);
    const float cr = exp(m - mn), pe = exp(s - mn);
    l = l * cr + pe;
    for (int e = 0; e < E; e++) acc[e] = acc[e] * cr + pe * float(v[e]);
    m = mn;
}

// Merges N partial (max, sum, acc) results in order 0..N-1 and writes the output row.
template <int E, int N, typename P>
inline void attn_merge(P tm, P tl, P ta, uint lane, device float* op) {
    float M = -1e30f;
    for (int s = 0; s < N; s++) M = max(M, tm[s]);
    float Ls = 0.0f;
    float o[E];
    for (int e = 0; e < E; e++) o[e] = 0.0f;
    for (int s = 0; s < N; s++) {
        const float f = exp(tm[s] - M);
        Ls += tl[s] * f;
        for (int e = 0; e < E; e++) o[e] += ta[s * E * 32 + e * 32 + lane] * f;
    }
    const float inv = 1.0f / Ls;
    for (int e = 0; e < E; e++) op[e * 32 + lane] = o[e] * inv;
}

template <int E, int ANSG, bool CAP>
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
    for (int j0 = attn_first_key(p, L - 1) + int(sg); j0 < L; j0 += ANSG * AU) {
        half kh[AU][E], vh[AU][E];
        for (int u = 0; u < AU; u++) {
            const int j = kv_slot(min(j0 + u * ANSG, L - 1), p.maxSeq);   // clamped: rows past L are loaded but unused
            device const half* kp = kb + ulong(j) * ulong(p.D);
            device const half* vp = vb + ulong(j) * ulong(p.D);
            for (int e = 0; e < E; e++) { kh[u][e] = kp[e * 32 + lane]; vh[u][e] = vp[e * 32 + lane]; }
        }
        for (int u = 0; u < AU; u++)
            if (j0 + u * ANSG < L) attn_update<E, CAP>(qr, acc, m, l, kh[u], vh[u], p.softcap);
    }
    threadgroup float tm[ANSG], tl[ANSG];
    threadgroup float ta[ANSG * E * 32];
    if (lane == 0) { tm[sg] = m; tl[sg] = l; }
    for (int e = 0; e < E; e++) ta[sg * E * 32 + e * 32 + lane] = acc[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) attn_merge<E, ANSG>((threadgroup const float*)tm, (threadgroup const float*)tl,
                                     (threadgroup const float*)ta, lane, out + t * p.outStride + h * p.D);
}

// Long-context decode (T = 1), split across threadgroups. The work is the
// `attention<E, ASPLIT_V>` kernel's, re-arranged without changing any result:
// virtual simdgroup v = tg.y * ASPLIT_NSG + sg takes keys v, v+ASPLIT_V, ...
// for G query heads of KV head tg.x (each K/V row is read once per GQA
// group), and writes its partial (max, sum, acc) per head. attention_combine
// then merges the ASPLIT_V partials per head in the same order. Large heads
// (256/512, Gemma) take their query heads in sub-groups of G: tg.z selects the
// sub-group, so the registers stay bounded and each K/V row is read once per sub-group.
// Partials layout (floats): pm[nH*V] | pl[nH*V] | pa[nH*V*D], row = h*V + v.
#define ASPLIT_V 32
#define ASPLIT_NSG 4
template <int E, int G, bool CAP>
kernel void attention_split(device const float* qkv  [[buffer(0)]],
                            device const half*  kc   [[buffer(1)]],
                            device const half*  vc   [[buffer(2)]],
                            device float*       part [[buffer(3)]],
                            constant AttnParams& p   [[buffer(4)]],
                            uint3 tg   [[threadgroup_position_in_grid]],
                            uint  sg   [[simdgroup_index_in_threadgroup]],
                            uint  lane [[thread_index_in_simdgroup]]) {
    const int kvh = int(tg.x);
    const int v = int(tg.y) * ASPLIT_NSG + int(sg);
    const int h0 = kvh * (p.nH / p.nKV) + int(tg.z) * G;
    float qr[G][E], acc[G][E], m[G], l[G];
    for (int g = 0; g < G; g++) {
        device const float* q = qkv + (h0 + g) * p.D;
        for (int e = 0; e < E; e++) { qr[g][e] = q[e * 32 + lane] * p.scale; acc[g][e] = 0.0f; }
        m[g] = -1e30f; l[g] = 0.0f;
    }
    const int L = p.offset + 1;
    device const half* kb = kc + ulong(kvh) * ulong(p.maxSeq) * ulong(p.D);
    device const half* vb = vc + ulong(kvh) * ulong(p.maxSeq) * ulong(p.D);
    for (int j0 = attn_first_key(p, L - 1) + v; j0 < L; j0 += ASPLIT_V * AU) {
        half kh[AU][E], vh[AU][E];
        for (int u = 0; u < AU; u++) {
            const int j = kv_slot(min(j0 + u * ASPLIT_V, L - 1), p.maxSeq);
            device const half* kp = kb + ulong(j) * ulong(p.D);
            device const half* vp = vb + ulong(j) * ulong(p.D);
            for (int e = 0; e < E; e++) { kh[u][e] = kp[e * 32 + lane]; vh[u][e] = vp[e * 32 + lane]; }
        }
        for (int u = 0; u < AU; u++)
            if (j0 + u * ASPLIT_V < L)
                for (int g = 0; g < G; g++) attn_update<E, CAP>(qr[g], acc[g], m[g], l[g], kh[u], vh[u], p.softcap);
    }
    // Virtual simdgroups with no keys still write (-1e30, 0, 0), exactly like the fused kernel.
    const int rows = p.nH * ASPLIT_V;
    for (int g = 0; g < G; g++) {
        const int r = (h0 + g) * ASPLIT_V + v;
        if (lane == 0) { part[r] = m[g]; part[rows + r] = l[g]; }
        device float* pa = part + 2 * rows + r * p.D;
        for (int e = 0; e < E; e++) pa[e * 32 + lane] = acc[g][e];
    }
}

// One simdgroup per head: merge the ASPLIT_V partials of attention_split.
template <int E>
kernel void attention_combine(device const float* part [[buffer(0)]],
                              device float*       out  [[buffer(1)]],
                              constant AttnParams& p   [[buffer(2)]],
                              uint h    [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const int rows = p.nH * ASPLIT_V;
    const int r = int(h) * ASPLIT_V;
    attn_merge<E, ASPLIT_V>((device const float*)(part + r), (device const float*)(part + rows + r),
                            (device const float*)(part + 2 * rows + r * p.D), lane, out + int(h) * p.D);
}

// Prefill attention: one threadgroup per (head, block of QB queries). Each K/V
// row is loaded once and scored against all QB queries; 8 simdgroups split the
// keys with per-query online softmax, merged through threadgroup memory.
template <int E, int QB, bool CAP>
kernel void attention_prefill(device const float* qkv [[buffer(0)]],
                              device const half*  kc  [[buffer(1)]],
                              device const half*  vc  [[buffer(2)]],
                              device float*       out [[buffer(3)]],
                              constant AttnParams& p  [[buffer(4)]],
                              uint2 tg   [[threadgroup_position_in_grid]],
                              uint  tid  [[thread_index_in_threadgroup]],
                              uint  sg   [[simdgroup_index_in_threadgroup]],
                              uint  lane [[thread_index_in_simdgroup]]) {
    const int NSG = 8;
    const int h = int(tg.x), t0 = int(tg.y) * QB;
    const int nq = min(QB, p.T - t0);
    const int kvh = h / (p.nH / p.nKV);
    float qr[QB][E], acc[QB][E], m[QB], l[QB];
    for (int qi = 0; qi < QB; qi++) {
        m[qi] = -1e30f; l[qi] = 0.0f;
        device const float* q = qkv + (t0 + min(qi, nq - 1)) * p.qStride + h * p.D;
        for (int e = 0; e < E; e++) { qr[qi][e] = q[e * 32 + lane] * p.scale; acc[qi][e] = 0.0f; }
    }
    const int L = p.offset + t0 + nq;
    device const half* kb = kc + ulong(kvh) * ulong(p.maxSeq) * ulong(p.D);
    device const half* vb = vc + ulong(kvh) * ulong(p.maxSeq) * ulong(p.D);
    const int jlo = attn_first_key(p, p.offset + t0);   // the first query's first key
    for (int j = jlo + int(sg); j < L; j += NSG) {
        float kv[E], vv[E];
        device const half* kp = kb + ulong(kv_slot(j, p.maxSeq)) * ulong(p.D);
        device const half* vp = vb + ulong(kv_slot(j, p.maxSeq)) * ulong(p.D);
        for (int e = 0; e < E; e++) { kv[e] = float(kp[e * 32 + lane]); vv[e] = float(vp[e * 32 + lane]); }
        for (int qi = 0; qi < QB; qi++) {
            if (qi >= nq || j > p.offset + t0 + qi) continue;   // causal mask (uniform per simdgroup)
            if (j < attn_first_key(p, p.offset + t0 + qi)) continue;   // sliding window
            float s = 0.0f;
            for (int e = 0; e < E; e++) s += qr[qi][e] * kv[e];
            s = attn_cap<CAP>(simd_sum(s), p.softcap);
            const float mn = max(m[qi], s);
            const float cr = exp(m[qi] - mn), pe = exp(s - mn);
            l[qi] = l[qi] * cr + pe;
            for (int e = 0; e < E; e++) acc[qi][e] = acc[qi][e] * cr + pe * vv[e];
            m[qi] = mn;
        }
    }
    threadgroup float tm[NSG][QB], tl[NSG][QB];
    threadgroup float o[QB][E * 32];
    if (lane == 0) for (int qi = 0; qi < QB; qi++) { tm[sg][qi] = m[qi]; tl[sg][qi] = l[qi]; }
    for (int i = int(tid); i < QB * E * 32; i += NSG * 32) o[i / (E * 32)][i % (E * 32)] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M[QB], Ls[QB];
    for (int qi = 0; qi < QB; qi++) {
        M[qi] = -1e30f;
        for (int s = 0; s < NSG; s++) M[qi] = max(M[qi], tm[s][qi]);
        Ls[qi] = 0.0f;
        for (int s = 0; s < NSG; s++) Ls[qi] += tl[s][qi] * exp(tm[s][qi] - M[qi]);
    }
    for (int s = 0; s < NSG; s++) {
        if (int(sg) == s) {
            for (int qi = 0; qi < nq; qi++) {
                const float f = exp(m[qi] - M[qi]);
                for (int e = 0; e < E; e++) o[qi][e * 32 + lane] += acc[qi][e] * f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int qi = int(sg); qi < nq; qi += NSG) {
        device float* op = out + (t0 + qi) * p.outStride + h * p.D;
        const float inv = 1.0f / Ls[qi];
        for (int e = 0; e < E; e++) op[e * 32 + lane] = o[qi][e * 32 + lane] * inv;
    }
}

// Flash attention for prefill on simdgroup matrices.
// Threadgroup = (head, 32 queries); 4 simdgroups × 8 query rows; key blocks of BK
// (16, or 8 for head_dim 256 so the K/V block fits threadgroup memory).
// S = Q·Kᵀ and O += P·V use 8×8 MMA; the online-softmax rescale of O is a
// multiply by a diagonal matrix. DD = head_dim / 8.
template <int DD, int BK, bool CAP>
kernel void flash_prefill(device const float* qkv [[buffer(0)]],
                          device const half*  kc  [[buffer(1)]],
                          device const half*  vc  [[buffer(2)]],
                          device float*       out [[buffer(3)]],
                          constant AttnParams& p  [[buffer(4)]],
                          uint2 tg   [[threadgroup_position_in_grid]],
                          uint  tid  [[thread_index_in_threadgroup]],
                          uint  sg   [[simdgroup_index_in_threadgroup]],
                          uint  lane [[thread_index_in_simdgroup]]) {
    constexpr int D = DD * 8;
    constexpr int NB = BK / 8;                   // 8×8 score tiles per key block
    constexpr int CPL = BK / 4;                  // score columns per lane (4 lanes per row)
    threadgroup float KV[2 * BK * D];            // K block then V block; reused for the output
    threadgroup float Ssh[4][8 * BK];
    threadgroup float Dg[4][64];
    threadgroup float* Ks = KV;
    threadgroup float* Vs = KV + BK * D;

    const int h = int(tg.x), t0 = int(tg.y) * 32;
    const int kvh = h / (p.nH / p.nKV);
    const int qRow0 = t0 + int(sg) * 8;           // first query (batch index) of this simdgroup
    const int lastQ = min(t0 + 32, p.T) - 1;
    const int L = p.offset + lastQ + 1;           // keys needed by this block

    simdgroup_matrix<float, 8, 8> Q[DD], O[DD];
    for (int d = 0; d < DD; d++) {
        simdgroup_load(Q[d], qkv + qRow0 * p.qStride + h * D + d * 8, p.qStride);
        O[d] = simdgroup_matrix<float, 8, 8>(0);
    }
    for (int i = int(lane); i < 64; i += 32) Dg[sg][i] = 0.0f;

    const int r = int(lane) / 4, c0 = (int(lane) % 4) * CPL;   // softmax ownership: row r, CPL columns
    const int qAbs = p.offset + qRow0 + r;                     // absolute position of row r
    const bool rowValid = qRow0 + r < p.T;
    float m = -1e30f, l = 0.0f;

    device const half* kb = kc + ulong(kvh) * ulong(p.maxSeq) * ulong(D);
    device const half* vb = vc + ulong(kvh) * ulong(p.maxSeq) * ulong(D);

    const int qFirst = attn_first_key(p, p.offset + t0);       // first key of the block's first query
    for (int j0 = qFirst / BK * BK; j0 < L; j0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // D is a power of two: row / column from i are shifts; the ring slot costs a modulo only
        // for positions past the cache length.
        for (int i = int(tid); i < BK * D; i += 128) {
            const int r = i / D, j = j0 + r;
            const bool ok = j < L;
            const ulong src = ulong(kv_slot(j, p.maxSeq)) * D + (i - r * D);
            Ks[i] = ok ? float(kb[src]) : 0.0f;
            Vs[i] = ok ? float(vb[src]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Skip key blocks that are entirely in the future for all 8 rows of this simdgroup.
        if (j0 > p.offset + qRow0 + 7) continue;

        simdgroup_matrix<float, 8, 8> S[NB], Kt;
        for (int b = 0; b < NB; b++) S[b] = simdgroup_matrix<float, 8, 8>(0);
        for (int d = 0; d < DD; d++) {
            for (int b = 0; b < NB; b++) {
                simdgroup_load(Kt, Ks + b * 8 * D + d * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[b], Q[d], Kt, S[b]);
            }
        }
        for (int b = 0; b < NB; b++) simdgroup_store(S[b], &Ssh[sg][b * 8], BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float s[CPL];
        float bmax = -1e30f;
        const int jMin = attn_first_key(p, qAbs);
        for (int c = 0; c < CPL; c++) {
            const int j = j0 + c0 + c;
            s[c] = (rowValid && j <= qAbs && j >= jMin && j < L) ? attn_cap<CAP>(Ssh[sg][r * BK + c0 + c] * p.scale, p.softcap) : -1e30f;
            bmax = max(bmax, s[c]);
        }
        bmax = max(bmax, simd_shuffle_xor(bmax, 1));
        bmax = max(bmax, simd_shuffle_xor(bmax, 2));
        const float mn = max(m, bmax);
        float bsum = 0.0f;
        for (int c = 0; c < CPL; c++) {
            const float pe = s[c] <= -1e29f ? 0.0f : exp(s[c] - mn);
            Ssh[sg][r * BK + c0 + c] = pe;
            bsum += pe;
        }
        bsum += simd_shuffle_xor(bsum, 1);
        bsum += simd_shuffle_xor(bsum, 2);
        const float corr = exp(m - mn);
        l = l * corr + bsum;
        m = mn;
        if (c0 == 0) Dg[sg][r * 8 + r] = corr;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> Dm, P[NB], Vb;
        simdgroup_load(Dm, &Dg[sg][0], 8);
        for (int b = 0; b < NB; b++) simdgroup_load(P[b], &Ssh[sg][b * 8], BK);
        for (int d = 0; d < DD; d++) {
            simdgroup_multiply(O[d], Dm, O[d]);
            for (int b = 0; b < NB; b++) {
                simdgroup_load(Vb, Vs + b * 8 * D + d * 8, D);
                simdgroup_multiply_accumulate(O[d], P[b], Vb, O[d]);
            }
        }
    }
    // Write out: stage each simdgroup's 8×D block in KV, normalise rows by l. KV holds
    // 2·BK·D floats: all four blocks at once for BK = 16, two simdgroups at a time for BK = 8.
    constexpr int STAGE = (2 * BK * D) / (8 * D);   // simdgroups staged per round (4 or 2)
    const float inv = l > 0 ? 1.0f / l : 0.0f;
    for (int round = 0; round < 4 / STAGE; round++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(sg) / STAGE != round) continue;
        threadgroup float* Osh = KV + (int(sg) % STAGE) * 8 * D;
        for (int d = 0; d < DD; d++) simdgroup_store(O[d], Osh + d * 8, D);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (rowValid) {
            device float* op = out + (qRow0 + r) * p.outStride + h * D;
            for (int c = int(lane) % 4; c < D; c += 4) op[c] = Osh[r * D + c] * inv;
        }
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
    out[t * p.cols + i] = act_mul(0, g, u);
}

// out[t, i] = gelu(gu[t, i]) * gu[t, cols + i]   (Gemma's GeGLU; tanh approximation)
kernel void gelu_tanh_mul(device const float* gu  [[buffer(0)]],
                          device float*       out [[buffer(1)]],
                          constant EWParams&  p   [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    const int i = int(gid.x), t = int(gid.y);
    if (i >= p.cols) return;
    const float g = gu[t * p.stride + i];
    const float u = gu[t * p.stride + p.cols + i];
    out[t * p.cols + i] = act_mul(1, g, u);
}

// out[t, i] = gelu(gu[t, i]) * gu[t, cols + i]   (exact GELU, Gemma 1)
kernel void gelu_mul(device const float* gu  [[buffer(0)]],
                     device float*       out [[buffer(1)]],
                     constant EWParams&  p   [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const int i = int(gid.x), t = int(gid.y);
    if (i >= p.cols) return;
    const float g = gu[t * p.stride + i];
    const float u = gu[t * p.stride + p.cols + i];
    out[t * p.cols + i] = act_mul(2, g, u);
}

// a[i] = (a[i] + b[i] * s.x) * s.y   (Gemma 4 per-layer inputs)
kernel void add_scaled(device float*       a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       constant EWParams&  p [[buffer(2)]],
                       constant float2&    s [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (int(gid) < p.n) a[gid] = (a[gid] + b[gid] * s.x) * s.y;
}

// g[t, i] = gelu(g[t, i]) * x[t * stride + i], in place over cols columns (Gemma 4 per-layer input gate)
kernel void gelu_tanh_gate(device float*       g [[buffer(0)]],
                           device const float* x [[buffer(1)]],
                           constant EWParams&  p [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    const int i = int(gid.x), t = int(gid.y);
    if (i >= p.cols) return;
    const float a = g[t * p.cols + i];
    g[t * p.cols + i] = act_mul(1, a, x[t * p.stride + i]);
}

// a[i] *= s   (Gemma's embedding scale)
kernel void scale_inplace(device float*       a [[buffer(0)]],
                          constant EWParams&  p [[buffer(1)]],
                          constant float&     s [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (int(gid) < p.n) a[gid] *= s;
}

// a[i] = tanh(a[i] / cap) * cap   (final-logit soft-capping)
kernel void softcap_inplace(device float*       a   [[buffer(0)]],
                            constant EWParams&  p   [[buffer(1)]],
                            constant float&     cap [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (int(gid) < p.n) a[gid] = precise::tanh(a[gid] / cap) * cap;
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

// ---------------------------------------------------------------------------
// Token selection on the GPU, in the same command buffer as the forward pass,
// so the next decode step can be queued before this one has finished and the
// GPU never waits for the CPU. One threadgroup: repetition penalty, then argmax
// (greedy) or exact top-k → softmax → min-p → top-p → one xorshift draw. The
// token goes to hist[pos + 1], which the next step's embedding lookup reads.
// ---------------------------------------------------------------------------
struct SampleParams { int V; int pos; int window; int topK; float penalty; float temperature; float topP; float minP; };

#define SMP_CAP 1024     // candidate pool (the exact top-k is sorted out of at most this many logits)

// Order-preserving float → uint key (larger value → larger key) and back.
inline uint smp_key(float f) { const uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
inline float smp_val(ulong key) { const uint u = uint(key >> 32); return as_type<float>((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u); }
inline int smp_id(ulong key) { return int(~uint(key)); }

kernel void sample_token(device float*          logits [[buffer(0)]],
                         device int*            hist   [[buffer(1)]],
                         device ulong*          rng    [[buffer(2)]],   // [0] xorshift64 state, then the pool-width hint (float)
                         device int*            status [[buffer(3)]],   // status[pos + 1]: 0 ok, 1 non-finite logits
                         constant SampleParams& p      [[buffer(4)]],
                         device int*            diag   [[buffer(5)]],   // [0] pool size, [1] attempts (profiling)
                         uint tid  [[thread_index_in_threadgroup]],
                         uint tsz  [[threads_per_threadgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint nsg  [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float shf[32];
    threadgroup int   shi[32];
    threadgroup ulong cand[SMP_CAP];
    threadgroup ulong sorted[SMP_CAP];
    threadgroup float prob[SMP_CAP];
    threadgroup atomic_uint np;
    const int V = p.V, pos = p.pos;

    // 1. Repetition penalty, once per distinct token among the last `window` of hist[0...pos].
    if (p.penalty != 1.0f) {
        const int n = min(p.window, pos + 1);
        for (int j = int(tid); j < n; j += int(tsz)) {
            const int t = hist[pos - j];
            bool dup = false;
            for (int j2 = 0; j2 < j; j2++) dup = dup || (hist[pos - j2] == t);
            if (!dup && t >= 0 && t < V) { const float v = logits[t]; logits[t] = v > 0.0f ? v / p.penalty : v * p.penalty; }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    // 2. Max logit and its lowest index. Passes over the logits take 8 contiguous values per lane
    //    (two vector loads in flight) so they are not bound by load latency.
    const int V8 = V & ~7;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i0 = int(tid) * 8; i0 < V8; i0 += int(tsz) * 8) {
        const float4 a = *(device const float4*)(logits + i0), b = *(device const float4*)(logits + i0 + 4);
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        for (int j = 0; j < 8; j++) if (v[j] > best) { best = v[j]; bi = i0 + j; }
    }
    for (int i = V8 + int(tid); i < V; i += int(tsz)) { const float v = logits[i]; if (v > best) { best = v; bi = i; } }
    const float sm = simd_max(best);
    const int si = simd_min(best == sm ? bi : 0x7fffffff);
    if (lane == 0) { shf[sgid] = sm; shi[sgid] = si; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        const float v = lane < nsg ? shf[lane] : -INFINITY;
        const int j = lane < nsg ? shi[lane] : 0x7fffffff;
        const float mm = simd_max(v);
        const int ii = simd_min(v == mm ? j : 0x7fffffff);
        if (lane == 0) { shf[0] = mm; shi[0] = ii; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float M = shf[0];
    const int argmax = shi[0];
    if (!isfinite(M)) { if (tid == 0) { hist[pos + 1] = 0; status[pos + 1] = 1; diag[0] = 0; diag[1] = 0; } return; }
    if (p.temperature <= 0.001f) { if (tid == 0) { hist[pos + 1] = argmax; status[pos + 1] = 0; diag[0] = 0; diag[1] = 0; } return; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Pool = { x >= M - delta } with k <= |pool| <= SMP_CAP, so it holds the exact top k. delta
    //    starts from the width that worked last step (kept in the state buffer) and is bisected on
    //    a miss; each attempt is one pass with one atomic per 32 logits. If more than the cap tie
    //    at the maximum (delta = 0), an index bound is bisected instead, so the lowest ids win.
    //    k is capped at SMP_CAP (the CPU sampler caps a disabled top-k at 512 for the same reason).
    const int k = min(p.topK > 0 ? min(p.topK, V) : min(512, V), SMP_CAP);
    // Aim for a pool of at most ~2k so sorting it stays cheap (SMP_CAP is the hard limit).
    const uint cap = uint(min(SMP_CAP, max(2 * k, 128)));
    device float* hint = (device float*)(rng + 1);
    float delta = hint[0];
    if (!(delta > 0.0f) || !(delta < 1.0e6f)) delta = 8.0f;
    float loD = 0.0f, hiD = INFINITY;   // |pool| < k at loD, > cap at hiD
    int limitB = V, loB = 0, hiB = V;   // index bound, used only once delta has reached 0
    uint count = 0;
    int attempts = 0;
    for (int attempt = 0; attempt < 64; attempt++) {
        attempts++;
        if (tid == 0) atomic_store_explicit(&np, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float thr = M - delta;
        // 256 logits per simdgroup step (8 per lane); one atomic reserves the step's pool slots.
        for (int base = int(sgid) * 256; base < V8; base += int(nsg) * 256) {
            const int i0 = base + int(lane) * 8;
            float v[8];
            if (i0 < V8) {
                const float4 a = *(device const float4*)(logits + i0), b = *(device const float4*)(logits + i0 + 4);
                v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
            } else {
                for (int j = 0; j < 8; j++) v[j] = -INFINITY;
            }
            uint c = 0;
            for (int j = 0; j < 8; j++) c += (v[j] >= thr && i0 + j < limitB) ? 1u : 0u;
            const uint total = simd_sum(c);
            if (total == 0u) continue;
            uint slot = 0;
            if (lane == 0) slot = atomic_fetch_add_explicit(&np, total, memory_order_relaxed);
            slot = simd_broadcast(slot, 0) + simd_prefix_exclusive_sum(c);
            for (int j = 0; j < 8; j++) {
                if (v[j] >= thr && i0 + j < limitB) {
                    if (slot < SMP_CAP) cand[slot] = (ulong(smp_key(v[j])) << 32) | ulong(~uint(i0 + j));
                    slot++;
                }
            }
        }
        for (int i = V8 + int(tid); i < V && i < limitB; i += int(tsz)) {
            const float v = logits[i];
            if (v >= thr) {
                const uint slot = atomic_fetch_add_explicit(&np, 1u, memory_order_relaxed);
                if (slot < SMP_CAP) cand[slot] = (ulong(smp_key(v)) << 32) | ulong(~uint(i));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        count = atomic_load_explicit(&np, memory_order_relaxed);
        if (count >= uint(k) && count <= cap) break;
        if (delta == 0.0f) {                                // ties at the max: bisect the index bound
            if (count > cap) hiB = limitB; else loB = limitB;
            limitB = (loB + hiB) / 2;
            if (limitB <= loB || limitB >= hiB) break;
        } else if (count > cap) {
            hiD = delta;
            delta = loD > 0.0f ? 0.5f * (loD + hiD) : 0.5f * delta;
            // The interval collapsed, or the threshold can no longer move: only the max level fits.
            if (delta <= loD || delta >= hiD || M - delta >= M) delta = 0.0f;
        } else {
            if (delta >= 1.0e6f) break;                     // fewer than k finite logits: take what there is
            loD = delta;
            delta = hiD < INFINITY ? 0.5f * (loD + hiD) : 4.0f * max(delta, 0.25f);
            if (delta <= loD || delta >= hiD) break;        // no float between the bounds: keep this pool
        }
    }
    if (tid == 0) { if (delta > 0.0f) hint[0] = delta; diag[0] = int(count); diag[1] = attempts; }
    const int pool = int(min(count, uint(SMP_CAP)));
    const int n = min(k, pool);

    // 5. Sort descending (value, then lowest index). Keys are distinct, so a rank sort (one pass
    //    per entry over the pool) is cheapest for the usual small pool; large pools use a bitonic
    //    network over the whole SMP_CAP array (unused slots hold 0, the smallest key).
    if (pool <= 256) {
        for (int i = int(tid); i < pool; i += int(tsz)) {
            const ulong me = cand[i];
            uint r = 0;
            for (int j = 0; j < pool; j++) r += cand[j] > me ? 1u : 0u;
            if (int(r) < n) sorted[r] = me;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (int i = pool + int(tid); i < SMP_CAP; i += int(tsz)) cand[i] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint size = 2; size <= SMP_CAP; size <<= 1) {
            for (uint stride = size >> 1; stride > 0; stride >>= 1) {
                for (uint q = tid; q < SMP_CAP / 2; q += tsz) {
                    const uint i = (q / stride) * (stride * 2) + (q % stride), j = i + stride;
                    const bool desc = (i & size) == 0;
                    const ulong a = cand[i], b = cand[j];
                    if ((a < b) == desc) { cand[i] = b; cand[j] = a; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (int i = int(tid); i < n; i += int(tsz)) sorted[i] = cand[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 6. Softmax over the top k, min-p and top-p cuts, one draw.
    const float invT = 1.0f / p.temperature;
    for (int i = int(tid); i < n; i += int(tsz)) prob[i] = exp((smp_val(sorted[i]) - M) * invT);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < n; i++) total += prob[i];
        int cut = n;
        if (p.minP > 0.0f) {
            const float th = p.minP * prob[0];
            for (int i = 0; i < n; i++) if (prob[i] < th) { cut = i; break; }
        }
        if (p.topP < 1.0f) {
            float acc = 0.0f;
            for (int i = 0; i < cut; i++) { acc += prob[i] / total; if (acc >= p.topP) { cut = i + 1; break; } }
        }
        cut = max(1, cut);
        float tot = 0.0f;
        for (int i = 0; i < cut; i++) tot += prob[i];
        ulong s = rng[0];
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        rng[0] = s;
        float r = float(s >> 40) * (1.0f / 16777216.0f);
        int chosen = smp_id(sorted[cut - 1]);
        for (int i = 0; i < cut; i++) { r -= prob[i] / tot; if (r <= 0.0f) { chosen = smp_id(sorted[i]); break; } }
        hist[pos + 1] = chosen; status[pos + 1] = 0;
    }
}

// Explicit instantiations ----------------------------------------------------
#define INST_QMM(B, T, TN, TT) \
  template [[host_name("qmm_" #B "_" #TN "_" #TT)]] kernel void qmm<B, T, TT>( \
    device const uchar*, device const T*, device const T*, device const float*, device float*, \
    constant MMParams&, uint2, uint, uint);
#define INST_QNEW(B, T, TN) \
  template [[host_name("qmv_" #B "_" #TN)]] kernel void qmv<B, T, 0, 0>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, device const float*, uint2, uint, uint); \
  template [[host_name("qmv_gate_" #B "_" #TN)]] kernel void qmv<B, T, 1, 0>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, device const float*, uint2, uint, uint); \
  template [[host_name("qgemm_" #B "_" #TN)]] kernel void qgemm<B, T>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint); \
  template [[host_name("qgemms_" #B "_" #TN "_8")]] kernel void qgemms<B, T, 8>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint); \
  template [[host_name("qgemms_" #B "_" #TN "_16")]] kernel void qgemms<B, T, 16>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint);
// 4-bit remainder variants (TAIL 4 / 8 / 16), plain and gated.
#define INST_QMV_TAIL(T, TN) \
  template [[host_name("qmv_t4_4_" #TN)]] kernel void qmv<4, T, 0, 4>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, device const float*, uint2, uint, uint); \
  template [[host_name("qmv_t8_4_" #TN)]] kernel void qmv<4, T, 0, 8>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, device const float*, uint2, uint, uint); \
  template [[host_name("qmv_t16_4_" #TN)]] kernel void qmv<4, T, 0, 16>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, device const float*, uint2, uint, uint); \
  template [[host_name("qmv_gate_t4_4_" #TN)]] kernel void qmv<4, T, 1, 4>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, device const float*, uint2, uint, uint); \
  template [[host_name("qmv_gate_t8_4_" #TN)]] kernel void qmv<4, T, 1, 8>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, device const float*, uint2, uint, uint); \
  template [[host_name("qmv_gate_t16_4_" #TN)]] kernel void qmv<4, T, 1, 16>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, device const float*, uint2, uint, uint);
INST_QMV_TAIL(half, f16) INST_QMV_TAIL(bfloat, bf16) INST_QMV_TAIL(float, f32)
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

#define QVBR 4
#define INST_QMVB(T, TN, TT) \
  template [[host_name("qmvb4_" #TN "_" #TT)]] kernel void qmvb4<T, TT, QVBR>(device const uchar*, device const T*, device const T*, device const float*, device float*, constant MMParams&, device const uchar*, device const T*, device const T*, device const uchar*, device const T*, device const T*, device const T*, device const T*, device const T*, uint2, uint, uint);
INST_QMVB(half, f16, 4) INST_QMVB(bfloat, bf16, 4) INST_QMVB(float, f32, 4)

#define INST_DENSE(T, TN) \
  template [[host_name("dmm_" #TN "_1")]] kernel void dmm<T, 1>(device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint); \
  template [[host_name("dmm_" #TN "_8")]] kernel void dmm<T, 8>(device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint); \
  template [[host_name("dgemm_" #TN)]] kernel void dgemm<T>(device const T*, device const float*, device float*, constant MMParams&, uint2, uint, uint); \
  template [[host_name("dembed_" #TN)]] kernel void dembed<T>(device const T*, device const int*, device float*, constant MMParams&, uint2); \
  template [[host_name("rms_norm_" #TN)]] kernel void rms_norm<T>(device const float*, device float*, device const T*, constant NormParams&, uint, uint, uint, uint, uint, uint); \
  template [[host_name("qkv_post_" #TN)]] kernel void qkv_post<T>(device float*, device const float*, device half*, device half*, device const T*, device const T*, constant QKVPostParams&, uint2, uint, uint, uint, uint); \
  template [[host_name("add_rms_norm_" #TN)]] kernel void add_rms_norm<T>(device float*, device const float*, device float*, device const T*, constant NormParams&, uint, uint, uint, uint, uint, uint); \
  template [[host_name("norm_add_rms_norm_" #TN)]] kernel void norm_add_rms_norm<T>(device float*, device const float*, device float*, device const T*, constant NormParams&, device const T*, constant float&, uint, uint, uint, uint, uint, uint); \
  template [[host_name("add_bias_" #TN)]] kernel void add_bias<T>(device float*, device const T*, constant EWParams&, uint2);

INST_DENSE(half, f16)
INST_DENSE(bfloat, bf16)
INST_DENSE(float, f32)

#define INST_ATTN_SG(E, SG, CAP, SFX) \
  template [[host_name("attention_" #E "_" #SG SFX)]] kernel void attention<E, SG, CAP>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint2, uint, uint);
#define INST_ATTN(E) INST_ATTN_SG(E, 8, false, "") INST_ATTN_SG(E, 16, false, "") INST_ATTN_SG(E, 32, false, "")
INST_ATTN(1)
INST_ATTN(2)
INST_ATTN(3)
INST_ATTN(4)
INST_ATTN(5)
INST_ATTN(6)
INST_ATTN(8)
// head_dim 512 (Gemma 4 global layers): 8 simdgroups keep the merge buffer within 32 KB.
INST_ATTN_SG(16, 8, false, "")
// Soft-capped variants for Gemma 2's head sizes (128 and 256).
INST_ATTN_SG(4, 32, true, "_cap")
INST_ATTN_SG(8, 16, true, "_cap")
#define INST_ASPLIT_C(E, G, CAP, SFX) \
  template [[host_name("attention_split_" #E "_" #G SFX)]] kernel void attention_split<E, G, CAP>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint3, uint, uint);
#define INST_ASPLIT(E, G) INST_ASPLIT_C(E, G, false, "")
INST_ASPLIT(2, 1) INST_ASPLIT(2, 2) INST_ASPLIT(2, 3) INST_ASPLIT(2, 4)
INST_ASPLIT(2, 5) INST_ASPLIT(2, 6) INST_ASPLIT(2, 7) INST_ASPLIT(2, 8)
INST_ASPLIT(3, 1) INST_ASPLIT(3, 2) INST_ASPLIT(3, 3) INST_ASPLIT(3, 4)
INST_ASPLIT(4, 1) INST_ASPLIT(4, 2) INST_ASPLIT(4, 3) INST_ASPLIT(4, 4)
INST_ASPLIT_C(4, 1, true, "_cap") INST_ASPLIT_C(4, 2, true, "_cap") INST_ASPLIT_C(4, 3, true, "_cap") INST_ASPLIT_C(4, 4, true, "_cap")
// head_dim 256 / 512 (Gemma): query heads in sub-groups of up to 4 / 2.
INST_ASPLIT(8, 1) INST_ASPLIT(8, 2) INST_ASPLIT(8, 4)
INST_ASPLIT(16, 1) INST_ASPLIT(16, 2)
INST_ASPLIT_C(8, 1, true, "_cap") INST_ASPLIT_C(8, 2, true, "_cap") INST_ASPLIT_C(8, 4, true, "_cap")
#define INST_ACOMB(E) \
  template [[host_name("attention_combine_" #E)]] kernel void attention_combine<E>(device const float*, device float*, constant AttnParams&, uint, uint);
INST_ACOMB(2) INST_ACOMB(3) INST_ACOMB(4) INST_ACOMB(8) INST_ACOMB(16)
#define INST_ATTNP_C(E, QB, CAP, SFX) \
  template [[host_name("attention_prefill_" #E SFX)]] kernel void attention_prefill<E, QB, CAP>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint2, uint, uint, uint);
#define INST_ATTNP(E, QB) INST_ATTNP_C(E, QB, false, "")
INST_ATTNP(1, 8)
INST_ATTNP(2, 8)
INST_ATTNP(3, 8)
INST_ATTNP(4, 8)
INST_ATTNP(5, 4)
INST_ATTNP(6, 4)
INST_ATTNP(8, 4)
INST_ATTNP(16, 2)
INST_ATTNP_C(8, 4, true, "_cap")
#define INST_FLASH_C(DD, BK, CAP, SFX) \
  template [[host_name("flash_prefill_" #DD SFX)]] kernel void flash_prefill<DD, BK, CAP>(device const float*, device const half*, device const half*, device float*, constant AttnParams&, uint2, uint, uint, uint);
#define INST_FLASH(DD) INST_FLASH_C(DD, 16, false, "")
INST_FLASH(4)
INST_FLASH(8)
INST_FLASH(12)
INST_FLASH(16)
INST_FLASH_C(16, 16, true, "_cap")
// head_dim 256 (Gemma): 8-key blocks keep the K/V block within threadgroup memory.
INST_FLASH_C(32, 8, false, "")
INST_FLASH_C(32, 8, true, "_cap")
"""#
}
