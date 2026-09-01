// PTX semantics differential test.
//
// Every case computes a value with a specific PTX construct and stores it.
// Running the SAME binary on a physical GPU and on VirtualGPU must produce
// identical bytes; any difference is a semantics bug in the interpreter.
// This is the oracle-based checking ARCHITECTURE.md describes, in miniature.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define N 256

__global__ void semantics(unsigned long long* out) {
    int t = threadIdx.x;
    unsigned long long r = 0;
    // Signed/unsigned shifts, including over-shift (PTX clamps).
    if (t == 0)  { int v = -16;      asm("shr.s32 %0, %1, %2;" : "=r"(v) : "r"(-16), "r"(2));  r = (unsigned)v; }
    if (t == 1)  { int v = 0;        asm("shr.s32 %0, %1, %2;" : "=r"(v) : "r"(-1),  "r"(40)); r = (unsigned)v; }
    if (t == 2)  { unsigned v = 0;   asm("shr.u32 %0, %1, %2;" : "=r"(v) : "r"(0xFFFFFFFFu), "r"(40)); r = v; }
    if (t == 3)  { int v = 0;        asm("shl.b32 %0, %1, %2;" : "=r"(v) : "r"(1), "r"(33));   r = (unsigned)v; }
    // Division and remainder with negative operands (truncation direction).
    if (t == 4)  { int v = 0;        asm("div.s32 %0, %1, %2;" : "=r"(v) : "r"(-7), "r"(2));   r = (unsigned)v; }
    if (t == 5)  { int v = 0;        asm("rem.s32 %0, %1, %2;" : "=r"(v) : "r"(-7), "r"(2));   r = (unsigned)v; }
    if (t == 6)  { int v = 0;        asm("rem.s32 %0, %1, %2;" : "=r"(v) : "r"(7),  "r"(-2));  r = (unsigned)v; }
    // min/max signedness.
    if (t == 7)  { int v = 0;        asm("min.s32 %0, %1, %2;" : "=r"(v) : "r"(-5), "r"(3));   r = (unsigned)v; }
    if (t == 8)  { unsigned v = 0;   asm("min.u32 %0, %1, %2;" : "=r"(v) : "r"(0xFFFFFFFBu), "r"(3u)); r = v; }
    // Wide multiply, signed and unsigned.
    if (t == 9)  { long long v = 0;  asm("mul.wide.s32 %0, %1, %2;" : "=l"(v) : "r"(-100000), "r"(100000)); r = (unsigned long long)v; }
    if (t == 10) { unsigned long long v=0; asm("mul.wide.u32 %0, %1, %2;" : "=l"(v) : "r"(0xFFFFFFFFu), "r"(2u)); r = v; }
    // mad.lo truncation.
    if (t == 11) { int v = 0;        asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(0x10000), "r"(0x10000), "r"(7)); r = (unsigned)v; }
    // Bitfield extract: signed, and a zero-length field.
    if (t == 12) { int v = 0;        asm("bfe.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(0xF0), "r"(4), "r"(4)); r = (unsigned)v; }
    if (t == 13) { int v = 0;        asm("bfe.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(-1), "r"(0), "r"(0)); r = (unsigned)v; }
    if (t == 14) { unsigned v = 0;   asm("bfe.u32 %0, %1, %2, %3;" : "=r"(v) : "r"(0xABCD1234u), "r"(40), "r"(8)); r = v; }
    // Bitfield insert with an out-of-range position.
    if (t == 15) { unsigned v = 0;   asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(v) : "r"(0xFFu), "r"(0u), "r"(40), "r"(8)); r = v; }
    // Funnel shifts.
    if (t == 16) { unsigned v = 0;   asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(0x9ABCDEF0u), "r"(0x12345678u), "r"(40u)); r = v; }
    if (t == 17) { unsigned v = 0;   asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(0x9ABCDEF0u), "r"(0x12345678u), "r"(40u)); r = v; }
    // Byte permute, including the sign-replicate selector.
    if (t == 18) { unsigned v = 0;   asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(0x03020100u), "r"(0x07060504u), "r"(0x0000BA98u)); r = v; }
    // popc / clz.
    if (t == 19) { unsigned v = 0;   asm("popc.b32 %0, %1;" : "=r"(v) : "r"(0xF0F0F0F0u)); r = v; }
    if (t == 20) { unsigned v = 0;   asm("clz.b32 %0, %1;"  : "=r"(v) : "r"(0u)); r = v; }
    if (t == 21) { unsigned v = 0;   asm("brev.b32 %0, %1;" : "=r"(v) : "r"(0x00000001u)); r = v; }
    // Float -> int conversions: rounding modes and saturation.
    if (t == 22) { int v = 0;        asm("cvt.rzi.s32.f32 %0, %1;" : "=r"(v) : "f"(-2.7f)); r = (unsigned)v; }
    if (t == 23) { int v = 0;        asm("cvt.rni.s32.f32 %0, %1;" : "=r"(v) : "f"(2.5f));  r = (unsigned)v; }
    if (t == 24) { int v = 0;        asm("cvt.rni.s32.f32 %0, %1;" : "=r"(v) : "f"(3.5f));  r = (unsigned)v; }
    if (t == 25) { int v = 0;        asm("cvt.rmi.s32.f32 %0, %1;" : "=r"(v) : "f"(-2.2f)); r = (unsigned)v; }
    if (t == 26) { int v = 0;        asm("cvt.rzi.s32.f32 %0, %1;" : "=r"(v) : "f"(1e30f)); r = (unsigned)v; }
    if (t == 27) { unsigned v = 0;   asm("cvt.rzi.u32.f32 %0, %1;" : "=r"(v) : "f"(-5.0f)); r = v; }
    // NaN-aware comparisons.
    if (t == 28) { unsigned v = 0; float nan = __int_as_float(0x7FC00000);
                   asm("{ .reg .pred %%p; setp.neu.f32 %%p, %1, %2; selp.u32 %0, 1, 0, %%p; }" : "=r"(v) : "f"(nan), "f"(1.0f)); r = v; }
    if (t == 29) { unsigned v = 0; float nan = __int_as_float(0x7FC00000);
                   asm("{ .reg .pred %%p; setp.gt.f32 %%p, %1, %2; selp.u32 %0, 1, 0, %%p; }" : "=r"(v) : "f"(nan), "f"(1.0f)); r = v; }
    // Float min/max with NaN, and signed zero.
    if (t == 30) { float v = 0; float nan = __int_as_float(0x7FC00000);
                   asm("min.f32 %0, %1, %2;" : "=f"(v) : "f"(nan), "f"(1.0f)); r = __float_as_uint(v); }
    if (t == 31) { float v = 0;      asm("max.f32 %0, %1, %2;" : "=f"(v) : "f"(-0.0f), "f"(0.0f)); r = __float_as_uint(v); }
    // Integer -> float rounding of a value that is not exactly representable.
    if (t == 32) { float v = 0;      asm("cvt.rn.f32.s32 %0, %1;" : "=f"(v) : "r"(16777217)); r = __float_as_uint(v); }
    if (t == 33) { float v = 0;      asm("cvt.rn.f32.u32 %0, %1;" : "=f"(v) : "r"(0xFFFFFFFFu)); r = __float_as_uint(v); }
    // f16 round-trip, including a value that rounds to even.
    if (t == 34) { unsigned short h = 0; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(3.14159f)); r = h; }
    if (t == 35) { unsigned short h = 0; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(65520.0f)); r = h; }
    if (t == 36) { unsigned short h = 0; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(1e-8f)); r = h; }
    if (t == 37) { float v = 0; unsigned short h = 0x3C01;
                   asm("cvt.f32.f16 %0, %1;" : "=f"(v) : "h"(h)); r = __float_as_uint(v); }
    // Warp shuffles must be executed by every thread in the member mask, so
    // these run unconditionally and only the store is selective. (Guarding the
    // shuffle itself with `if` deadlocks real hardware -- a good reminder that
    // the simulator must not be more permissive than the machine.)
    {
        unsigned down = 0, downp = 0, up = 0, bfly = 0;
        asm("{ .reg .pred %%q; shfl.sync.down.b32 %0|%%q, %2, %3, %4, %5; selp.u32 %1, 1, 0, %%q; }"
            : "=r"(down), "=r"(downp) : "r"(1000u + t), "r"(30u), "r"(31u), "r"(0xFFFFFFFFu));
        asm("shfl.sync.up.b32 %0, %1, %2, %3, %4;"
            : "=r"(up) : "r"(2000u + t), "r"(4u), "r"(0u), "r"(0xFFFFFFFFu));
        asm("shfl.sync.bfly.b32 %0, %1, %2, %3, %4;"
            : "=r"(bfly) : "r"(3000u + t), "r"(16u), "r"(31u), "r"(0xFFFFFFFFu));
        if (t >= 40 && t < 72) r = ((unsigned long long)downp << 32) | down;
        if (t >= 90 && t < 122) r = up;
        if (t >= 130 && t < 162) r = bfly;
    }
    // Transcendentals (approximate on hardware; compared with tolerance).
    if (t == 80) { float v = 0; asm("ex2.approx.f32 %0, %1;"   : "=f"(v) : "f"(3.0f));  r = __float_as_uint(v); }
    if (t == 81) { float v = 0; asm("lg2.approx.f32 %0, %1;"   : "=f"(v) : "f"(8.0f));  r = __float_as_uint(v); }
    if (t == 82) { float v = 0; asm("rsqrt.approx.f32 %0, %1;" : "=f"(v) : "f"(16.0f)); r = __float_as_uint(v); }
    if (t == 83) { float v = 0; asm("sin.approx.f32 %0, %1;"   : "=f"(v) : "f"(1.0f));  r = __float_as_uint(v); }
    if (t == 84) { float v = 0; asm("rcp.approx.f32 %0, %1;"   : "=f"(v) : "f"(3.0f));  r = __float_as_uint(v); }
    if (t == 85) { float v = 0; asm("sqrt.rn.f32 %0, %1;"      : "=f"(v) : "f"(2.0f));  r = __float_as_uint(v); }
    out[t] = r;
}

int main(int argc, char** argv) {
    unsigned long long *d, h[N];
    if (cudaMalloc(&d, sizeof h) != cudaSuccess) { printf("ALLOCFAIL\n"); return 1; }
    cudaMemset(d, 0, sizeof h);
    semantics<<<1, N>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("LAUNCHFAIL %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) printf("%d %016llx\n", i, h[i]);
    cudaFree(d);
    return 0;
}
