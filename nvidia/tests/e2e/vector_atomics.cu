// Vector atomics and reductions (PTX ISA 8.1, sm_90): atom/red .v2/.v4 on .f32,
// .v2/.v4/.v8 on .f16 and .bf16, .v2/.v4 on the packed .f16x2 and .bf16x2,
// with .add (and .min, .max on the half types), each with and without a
// result. Every form is run by 128 threads in four blocks onto the same
// words, so the contention checks the per-word atomicity too:
//   - add: every thread adds the same per-element addend, so the sum is exact
//     in any order (small integers, which f16 and bf16 hold) and the values the
//     atoms return are the partial sums, each exactly once;
//   - min, max: each thread brings a different value per element; the result
//     is the extreme, and every returned value lies between it and the start.
// Whether NaNs, subnormals and signed zeros combine as the PTX ISA says is not
// asserted here: it is derived from the documentation, not checked on a card.
// Prints PASS on the last line. Built for sm_90 and later (CUDA 12.1+).
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Form {
  const char* insn;
  const char* type;
  int vec;
  const char* op;
  int atom;
};
static const Form kForms[] = {
  {"red.global.add.v2.f32", "f32", 2, "add", 0},
  {"atom.global.add.v2.f32", "f32", 2, "add", 1},
  {"red.global.add.v4.f32", "f32", 4, "add", 0},
  {"atom.global.add.v4.f32", "f32", 4, "add", 1},
  {"red.global.add.noftz.v2.f16", "f16", 2, "add", 0},
  {"atom.global.add.noftz.v2.f16", "f16", 2, "add", 1},
  {"red.global.min.noftz.v2.f16", "f16", 2, "min", 0},
  {"atom.global.min.noftz.v2.f16", "f16", 2, "min", 1},
  {"red.global.max.noftz.v2.f16", "f16", 2, "max", 0},
  {"atom.global.max.noftz.v2.f16", "f16", 2, "max", 1},
  {"red.global.add.noftz.v4.f16", "f16", 4, "add", 0},
  {"atom.global.add.noftz.v4.f16", "f16", 4, "add", 1},
  {"red.global.min.noftz.v4.f16", "f16", 4, "min", 0},
  {"atom.global.min.noftz.v4.f16", "f16", 4, "min", 1},
  {"red.global.max.noftz.v4.f16", "f16", 4, "max", 0},
  {"atom.global.max.noftz.v4.f16", "f16", 4, "max", 1},
  {"red.global.add.noftz.v8.f16", "f16", 8, "add", 0},
  {"atom.global.add.noftz.v8.f16", "f16", 8, "add", 1},
  {"red.global.min.noftz.v8.f16", "f16", 8, "min", 0},
  {"atom.global.min.noftz.v8.f16", "f16", 8, "min", 1},
  {"red.global.max.noftz.v8.f16", "f16", 8, "max", 0},
  {"atom.global.max.noftz.v8.f16", "f16", 8, "max", 1},
  {"red.global.add.noftz.v2.bf16", "bf16", 2, "add", 0},
  {"atom.global.add.noftz.v2.bf16", "bf16", 2, "add", 1},
  {"red.global.min.noftz.v2.bf16", "bf16", 2, "min", 0},
  {"atom.global.min.noftz.v2.bf16", "bf16", 2, "min", 1},
  {"red.global.max.noftz.v2.bf16", "bf16", 2, "max", 0},
  {"atom.global.max.noftz.v2.bf16", "bf16", 2, "max", 1},
  {"red.global.add.noftz.v4.bf16", "bf16", 4, "add", 0},
  {"atom.global.add.noftz.v4.bf16", "bf16", 4, "add", 1},
  {"red.global.min.noftz.v4.bf16", "bf16", 4, "min", 0},
  {"atom.global.min.noftz.v4.bf16", "bf16", 4, "min", 1},
  {"red.global.max.noftz.v4.bf16", "bf16", 4, "max", 0},
  {"atom.global.max.noftz.v4.bf16", "bf16", 4, "max", 1},
  {"red.global.add.noftz.v8.bf16", "bf16", 8, "add", 0},
  {"atom.global.add.noftz.v8.bf16", "bf16", 8, "add", 1},
  {"red.global.min.noftz.v8.bf16", "bf16", 8, "min", 0},
  {"atom.global.min.noftz.v8.bf16", "bf16", 8, "min", 1},
  {"red.global.max.noftz.v8.bf16", "bf16", 8, "max", 0},
  {"atom.global.max.noftz.v8.bf16", "bf16", 8, "max", 1},
  {"red.global.add.noftz.v2.f16x2", "f16x2", 2, "add", 0},
  {"atom.global.add.noftz.v2.f16x2", "f16x2", 2, "add", 1},
  {"red.global.min.noftz.v2.f16x2", "f16x2", 2, "min", 0},
  {"atom.global.min.noftz.v2.f16x2", "f16x2", 2, "min", 1},
  {"red.global.max.noftz.v2.f16x2", "f16x2", 2, "max", 0},
  {"atom.global.max.noftz.v2.f16x2", "f16x2", 2, "max", 1},
  {"red.global.add.noftz.v4.f16x2", "f16x2", 4, "add", 0},
  {"atom.global.add.noftz.v4.f16x2", "f16x2", 4, "add", 1},
  {"red.global.min.noftz.v4.f16x2", "f16x2", 4, "min", 0},
  {"atom.global.min.noftz.v4.f16x2", "f16x2", 4, "min", 1},
  {"red.global.max.noftz.v4.f16x2", "f16x2", 4, "max", 0},
  {"atom.global.max.noftz.v4.f16x2", "f16x2", 4, "max", 1},
  {"red.global.add.noftz.v2.bf16x2", "bf16x2", 2, "add", 0},
  {"atom.global.add.noftz.v2.bf16x2", "bf16x2", 2, "add", 1},
  {"red.global.min.noftz.v2.bf16x2", "bf16x2", 2, "min", 0},
  {"atom.global.min.noftz.v2.bf16x2", "bf16x2", 2, "min", 1},
  {"red.global.max.noftz.v2.bf16x2", "bf16x2", 2, "max", 0},
  {"atom.global.max.noftz.v2.bf16x2", "bf16x2", 2, "max", 1},
  {"red.global.add.noftz.v4.bf16x2", "bf16x2", 4, "add", 0},
  {"atom.global.add.noftz.v4.bf16x2", "bf16x2", 4, "add", 1},
  {"red.global.min.noftz.v4.bf16x2", "bf16x2", 4, "min", 0},
  {"atom.global.min.noftz.v4.bf16x2", "bf16x2", 4, "min", 1},
  {"red.global.max.noftz.v4.bf16x2", "bf16x2", 4, "max", 0},
  {"atom.global.max.noftz.v4.bf16x2", "bf16x2", 4, "max", 1}
};
static const int kNumForms = 64;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1))
#define VGPU_VECTOR_ATOMICS 1
#endif

// b: per-thread operand registers (low 16 bits for the half types); o: what
// an atom returned in each register.
__global__ void run(int form, unsigned* mem, const unsigned* bv, unsigned* olds) {
#ifdef VGPU_VECTOR_ATOMICS
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned* b = bv + t * 8;
  unsigned* o = olds + t * 8;
  switch (form) {
    case 0: {
      asm volatile("red.global.add.v2.f32 [%0], {%1,%2};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      break;
    }
    case 1: {
      unsigned r0; unsigned r1;
      asm volatile("atom.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};" : "=r"(r0), "=r"(r1) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 2: {
      asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      break;
    }
    case 3: {
      unsigned r0; unsigned r1; unsigned r2; unsigned r3;
      asm volatile("atom.global.add.v4.f32 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 4: {
      asm volatile("red.global.add.noftz.v2.f16 [%0], {%1,%2};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      break;
    }
    case 5: {
      unsigned short r0; unsigned short r1;
      asm volatile("atom.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};" : "=h"(r0), "=h"(r1) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 6: {
      asm volatile("red.global.min.noftz.v2.f16 [%0], {%1,%2};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      break;
    }
    case 7: {
      unsigned short r0; unsigned short r1;
      asm volatile("atom.global.min.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};" : "=h"(r0), "=h"(r1) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 8: {
      asm volatile("red.global.max.noftz.v2.f16 [%0], {%1,%2};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      break;
    }
    case 9: {
      unsigned short r0; unsigned short r1;
      asm volatile("atom.global.max.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};" : "=h"(r0), "=h"(r1) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 10: {
      asm volatile("red.global.add.noftz.v4.f16 [%0], {%1,%2,%3,%4};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      break;
    }
    case 11: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3;
      asm volatile("atom.global.add.noftz.v4.f16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 12: {
      asm volatile("red.global.min.noftz.v4.f16 [%0], {%1,%2,%3,%4};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      break;
    }
    case 13: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3;
      asm volatile("atom.global.min.noftz.v4.f16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 14: {
      asm volatile("red.global.max.noftz.v4.f16 [%0], {%1,%2,%3,%4};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      break;
    }
    case 15: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3;
      asm volatile("atom.global.max.noftz.v4.f16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 16: {
      asm volatile("red.global.add.noftz.v8.f16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      break;
    }
    case 17: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3; unsigned short r4; unsigned short r5; unsigned short r6; unsigned short r7;
      asm volatile("atom.global.add.noftz.v8.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3), "=h"(r4), "=h"(r5), "=h"(r6), "=h"(r7) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3; o[4] = r4; o[5] = r5; o[6] = r6; o[7] = r7;
      break;
    }
    case 18: {
      asm volatile("red.global.min.noftz.v8.f16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      break;
    }
    case 19: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3; unsigned short r4; unsigned short r5; unsigned short r6; unsigned short r7;
      asm volatile("atom.global.min.noftz.v8.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3), "=h"(r4), "=h"(r5), "=h"(r6), "=h"(r7) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3; o[4] = r4; o[5] = r5; o[6] = r6; o[7] = r7;
      break;
    }
    case 20: {
      asm volatile("red.global.max.noftz.v8.f16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      break;
    }
    case 21: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3; unsigned short r4; unsigned short r5; unsigned short r6; unsigned short r7;
      asm volatile("atom.global.max.noftz.v8.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3), "=h"(r4), "=h"(r5), "=h"(r6), "=h"(r7) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3; o[4] = r4; o[5] = r5; o[6] = r6; o[7] = r7;
      break;
    }
    case 22: {
      asm volatile("red.global.add.noftz.v2.bf16 [%0], {%1,%2};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      break;
    }
    case 23: {
      unsigned short r0; unsigned short r1;
      asm volatile("atom.global.add.noftz.v2.bf16 {%0,%1}, [%2], {%3,%4};" : "=h"(r0), "=h"(r1) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 24: {
      asm volatile("red.global.min.noftz.v2.bf16 [%0], {%1,%2};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      break;
    }
    case 25: {
      unsigned short r0; unsigned short r1;
      asm volatile("atom.global.min.noftz.v2.bf16 {%0,%1}, [%2], {%3,%4};" : "=h"(r0), "=h"(r1) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 26: {
      asm volatile("red.global.max.noftz.v2.bf16 [%0], {%1,%2};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      break;
    }
    case 27: {
      unsigned short r0; unsigned short r1;
      asm volatile("atom.global.max.noftz.v2.bf16 {%0,%1}, [%2], {%3,%4};" : "=h"(r0), "=h"(r1) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 28: {
      asm volatile("red.global.add.noftz.v4.bf16 [%0], {%1,%2,%3,%4};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      break;
    }
    case 29: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3;
      asm volatile("atom.global.add.noftz.v4.bf16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 30: {
      asm volatile("red.global.min.noftz.v4.bf16 [%0], {%1,%2,%3,%4};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      break;
    }
    case 31: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3;
      asm volatile("atom.global.min.noftz.v4.bf16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 32: {
      asm volatile("red.global.max.noftz.v4.bf16 [%0], {%1,%2,%3,%4};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      break;
    }
    case 33: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3;
      asm volatile("atom.global.max.noftz.v4.bf16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 34: {
      asm volatile("red.global.add.noftz.v8.bf16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      break;
    }
    case 35: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3; unsigned short r4; unsigned short r5; unsigned short r6; unsigned short r7;
      asm volatile("atom.global.add.noftz.v8.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3), "=h"(r4), "=h"(r5), "=h"(r6), "=h"(r7) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3; o[4] = r4; o[5] = r5; o[6] = r6; o[7] = r7;
      break;
    }
    case 36: {
      asm volatile("red.global.min.noftz.v8.bf16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      break;
    }
    case 37: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3; unsigned short r4; unsigned short r5; unsigned short r6; unsigned short r7;
      asm volatile("atom.global.min.noftz.v8.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3), "=h"(r4), "=h"(r5), "=h"(r6), "=h"(r7) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3; o[4] = r4; o[5] = r5; o[6] = r6; o[7] = r7;
      break;
    }
    case 38: {
      asm volatile("red.global.max.noftz.v8.bf16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      break;
    }
    case 39: {
      unsigned short r0; unsigned short r1; unsigned short r2; unsigned short r3; unsigned short r4; unsigned short r5; unsigned short r6; unsigned short r7;
      asm volatile("atom.global.max.noftz.v8.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(r0), "=h"(r1), "=h"(r2), "=h"(r3), "=h"(r4), "=h"(r5), "=h"(r6), "=h"(r7) : "l"(mem), "h"((unsigned short)b[0]), "h"((unsigned short)b[1]), "h"((unsigned short)b[2]), "h"((unsigned short)b[3]), "h"((unsigned short)b[4]), "h"((unsigned short)b[5]), "h"((unsigned short)b[6]), "h"((unsigned short)b[7]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3; o[4] = r4; o[5] = r5; o[6] = r6; o[7] = r7;
      break;
    }
    case 40: {
      asm volatile("red.global.add.noftz.v2.f16x2 [%0], {%1,%2};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      break;
    }
    case 41: {
      unsigned r0; unsigned r1;
      asm volatile("atom.global.add.noftz.v2.f16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(r0), "=r"(r1) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 42: {
      asm volatile("red.global.min.noftz.v2.f16x2 [%0], {%1,%2};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      break;
    }
    case 43: {
      unsigned r0; unsigned r1;
      asm volatile("atom.global.min.noftz.v2.f16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(r0), "=r"(r1) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 44: {
      asm volatile("red.global.max.noftz.v2.f16x2 [%0], {%1,%2};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      break;
    }
    case 45: {
      unsigned r0; unsigned r1;
      asm volatile("atom.global.max.noftz.v2.f16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(r0), "=r"(r1) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 46: {
      asm volatile("red.global.add.noftz.v4.f16x2 [%0], {%1,%2,%3,%4};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      break;
    }
    case 47: {
      unsigned r0; unsigned r1; unsigned r2; unsigned r3;
      asm volatile("atom.global.add.noftz.v4.f16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 48: {
      asm volatile("red.global.min.noftz.v4.f16x2 [%0], {%1,%2,%3,%4};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      break;
    }
    case 49: {
      unsigned r0; unsigned r1; unsigned r2; unsigned r3;
      asm volatile("atom.global.min.noftz.v4.f16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 50: {
      asm volatile("red.global.max.noftz.v4.f16x2 [%0], {%1,%2,%3,%4};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      break;
    }
    case 51: {
      unsigned r0; unsigned r1; unsigned r2; unsigned r3;
      asm volatile("atom.global.max.noftz.v4.f16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 52: {
      asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1,%2};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      break;
    }
    case 53: {
      unsigned r0; unsigned r1;
      asm volatile("atom.global.add.noftz.v2.bf16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(r0), "=r"(r1) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 54: {
      asm volatile("red.global.min.noftz.v2.bf16x2 [%0], {%1,%2};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      break;
    }
    case 55: {
      unsigned r0; unsigned r1;
      asm volatile("atom.global.min.noftz.v2.bf16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(r0), "=r"(r1) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 56: {
      asm volatile("red.global.max.noftz.v2.bf16x2 [%0], {%1,%2};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      break;
    }
    case 57: {
      unsigned r0; unsigned r1;
      asm volatile("atom.global.max.noftz.v2.bf16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(r0), "=r"(r1) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]) : "memory");
      o[0] = r0; o[1] = r1;
      break;
    }
    case 58: {
      asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1,%2,%3,%4};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      break;
    }
    case 59: {
      unsigned r0; unsigned r1; unsigned r2; unsigned r3;
      asm volatile("atom.global.add.noftz.v4.bf16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 60: {
      asm volatile("red.global.min.noftz.v4.bf16x2 [%0], {%1,%2,%3,%4};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      break;
    }
    case 61: {
      unsigned r0; unsigned r1; unsigned r2; unsigned r3;
      asm volatile("atom.global.min.noftz.v4.bf16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    case 62: {
      asm volatile("red.global.max.noftz.v4.bf16x2 [%0], {%1,%2,%3,%4};" :: "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      break;
    }
    case 63: {
      unsigned r0; unsigned r1; unsigned r2; unsigned r3;
      asm volatile("atom.global.max.noftz.v4.bf16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(mem), "r"((unsigned)b[0]), "r"((unsigned)b[1]), "r"((unsigned)b[2]), "r"((unsigned)b[3]) : "memory");
      o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
      break;
    }
    default: break;
  }
#endif
}

static float half_val(const char* ty, unsigned short x) {
  return std::strncmp(ty, "bf16", 4) == 0 ? __bfloat162float(__ushort_as_bfloat16(x)) : __half2float(__ushort_as_half(x));
}
static unsigned short half_bits(const char* ty, float f) {
  return std::strncmp(ty, "bf16", 4) == 0 ? __bfloat16_as_ushort(__float2bfloat16_rn(f)) : __half_as_ushort(__float2half_rn(f));
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("FAIL %s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

int main() {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major < 9) { std::printf("SKIP: vector atomics need sm_90\nPASS\n"); return 0; }
  const int kBlocks = 4, kThreads = 32, kN = kBlocks * kThreads;
  unsigned *mem, *bv, *olds;
  CK(cudaMalloc(&mem, 64));
  CK(cudaMalloc(&bv, kN * 8 * 4));
  CK(cudaMalloc(&olds, kN * 8 * 4));
  int fails = 0;
  for (int f = 0; f < kNumForms; ++f) {
    const Form& F = kForms[f];
    const bool is32 = std::strcmp(F.type, "f32") == 0;
    const bool packed = std::strstr(F.type, "x2") != nullptr;
    const int regs = F.vec;                              // registers in the vector
    const int elems = is32 ? F.vec : packed ? 2 * F.vec : F.vec;   // values (halves, or floats)
    const int words = is32 || packed ? F.vec : F.vec / 2;
    const bool add = std::strcmp(F.op, "add") == 0, mn = std::strcmp(F.op, "min") == 0;
    // The start value of each element, and what thread t brings for it.
    auto init_of = [&](int) { return add ? 0.0f : mn ? 64.0f : -64.0f; };
    auto val_of = [&](int t, int e) {
      if (add) return 1.0f * static_cast<float>(1 << (e % 3));                   // 1, 2, 4
      return static_cast<float>(((t * (e + 3) + e) % 23) - 11);                  // -11 .. 11
    };
    std::vector<unsigned> hb(kN * 8, 0), hm(16, 0);
    for (int t = 0; t < kN; ++t)
      for (int r = 0; r < regs; ++r) {
        unsigned bits;
        if (is32) { float v = val_of(t, r); std::memcpy(&bits, &v, 4); }
        else if (!packed) bits = half_bits(F.type, val_of(t, r));
        else bits = half_bits(F.type, val_of(t, 2 * r)) | (static_cast<unsigned>(half_bits(F.type, val_of(t, 2 * r + 1))) << 16);
        hb[t * 8 + r] = bits;
      }
    for (int w = 0; w < words; ++w) {
      if (is32) { float v = init_of(w); std::memcpy(&hm[w], &v, 4); }
      else hm[w] = half_bits(F.type, init_of(2 * w)) | (static_cast<unsigned>(half_bits(F.type, init_of(2 * w + 1))) << 16);
    }
    CK(cudaMemcpy(mem, hm.data(), 64, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bv, hb.data(), kN * 8 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(olds, 0xff, kN * 8 * 4));
    run<<<kBlocks, kThreads>>>(f, mem, bv, olds);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<unsigned> gm(16), go(kN * 8);
    CK(cudaMemcpy(gm.data(), mem, 64, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(go.data(), olds, kN * 8 * 4, cudaMemcpyDeviceToHost));
    auto elem_of = [&](const std::vector<unsigned>& words_, int stride_base, int e) {
      // element e of a vector held as words (memory) or registers
      if (is32) { float v; std::memcpy(&v, &words_[stride_base + e], 4); return v; }
      const unsigned w = words_[stride_base + e / 2];
      return half_val(F.type, static_cast<unsigned short>((w >> (16 * (e % 2))) & 0xffff));
    };
    std::string why;
    for (int e = 0; e < elems && why.empty(); ++e) {
      float want = init_of(e);
      for (int t = 0; t < kN; ++t) {
        const float v = val_of(t, e);
        want = add ? want + v : mn ? std::min(want, v) : std::max(want, v);
      }
      const float got = elem_of(gm, 0, e);
      if (got != want) why = "element " + std::to_string(e) + " ends as " + std::to_string(got) + ", wanted " + std::to_string(want);
    }
    if (why.empty() && F.atom) {
      for (int e = 0; e < elems && why.empty(); ++e) {
        std::vector<float> seen;
        // Registers: a half type's register e holds element e's old value; a
        // packed or f32 register holds a word (element 2r and 2r+1 packed).
        for (int t = 0; t < kN; ++t) {
          float v;
          if (is32) { std::memcpy(&v, &go[t * 8 + e], 4); }
          else if (!packed) v = half_val(F.type, static_cast<unsigned short>(go[t * 8 + e] & 0xffff));
          else v = half_val(F.type, static_cast<unsigned short>((go[t * 8 + e / 2] >> (16 * (e % 2))) & 0xffff));
          seen.push_back(v);
        }
        std::sort(seen.begin(), seen.end());
        if (add) {
          // The partial sums, each exactly once: start + k * addend.
          for (int k = 0; k < kN && why.empty(); ++k)
            if (seen[k] != init_of(e) + k * val_of(0, e)) why = "element " + std::to_string(e) + ": the values atom returned are not the partial sums";
        } else {
          const float fin = elem_of(gm, 0, e);
          for (float s : seen)
            if (mn ? (s < fin || s > 64.0f) : (s > fin || s < -64.0f)) { why = "element " + std::to_string(e) + ": an atom returned " + std::to_string(s) + " outside [final, start]"; break; }
        }
      }
    }
    if (why.empty()) {
      std::printf("ok   %s\n", F.insn);
    } else {
      std::printf("FAIL %s: %s\n", F.insn, why.c_str());
      ++fails;
    }
  }
  cudaFree(mem);
  cudaFree(bv);
  cudaFree(olds);
  std::puts(fails ? "FAIL" : "PASS");
  return fails != 0;
}
