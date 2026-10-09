// Decoder probe: sm_90's vector atomics and reductions (atom/red .v2/.v4/.v8 on
// f32, f16, bf16 and the packed f16x2/bf16x2), every operation they take, and
// with and without a result. sm_90 and later.
__global__ void k(unsigned* mem, const unsigned* in) {
  unsigned v[8];
  for (int i = 0; i < 8; ++i) v[i] = in[threadIdx.x * 8 + i];
  unsigned r[8];
  unsigned* a = mem + threadIdx.x * 32;
  asm volatile("red.global.add.v2.f32 [%0], {%1,%2};" :: "l"(a), "f"(__uint_as_float(v[0])), "f"(__uint_as_float(v[1])) : "memory");
  { float o0; float o1;
    asm volatile("atom.global.add.v2.f32 {%0,%1}, [%2], {%3,%4};" : "=f"(o0), "=f"(o1) : "l"(a), "f"(__uint_as_float(v[0])), "f"(__uint_as_float(v[1])) : "memory");
    r[0] = __float_as_uint(o0); r[1] = __float_as_uint(o1); }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(a), "f"(__uint_as_float(v[0])), "f"(__uint_as_float(v[1])), "f"(__uint_as_float(v[2])), "f"(__uint_as_float(v[3])) : "memory");
  { float o0; float o1; float o2; float o3;
    asm volatile("atom.global.add.v4.f32 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=f"(o0), "=f"(o1), "=f"(o2), "=f"(o3) : "l"(a), "f"(__uint_as_float(v[0])), "f"(__uint_as_float(v[1])), "f"(__uint_as_float(v[2])), "f"(__uint_as_float(v[3])) : "memory");
    r[0] = __float_as_uint(o0); r[1] = __float_as_uint(o1); r[2] = __float_as_uint(o2); r[3] = __float_as_uint(o3); }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v2.f16 [%0], {%1,%2};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
  { unsigned short o0; unsigned short o1;
    asm volatile("atom.global.add.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};" : "=h"(o0), "=h"(o1) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v4.f16 [%0], {%1,%2,%3,%4};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3;
    asm volatile("atom.global.add.noftz.v4.f16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v8.f16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3; unsigned short o4; unsigned short o5; unsigned short o6; unsigned short o7;
    asm volatile("atom.global.add.noftz.v8.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3), "=h"(o4), "=h"(o5), "=h"(o6), "=h"(o7) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; r[4] = o4; r[5] = o5; r[6] = o6; r[7] = o7; }
  for (int i = 0; i < 8; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v2.bf16 [%0], {%1,%2};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
  { unsigned short o0; unsigned short o1;
    asm volatile("atom.global.add.noftz.v2.bf16 {%0,%1}, [%2], {%3,%4};" : "=h"(o0), "=h"(o1) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v4.bf16 [%0], {%1,%2,%3,%4};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3;
    asm volatile("atom.global.add.noftz.v4.bf16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v8.bf16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3; unsigned short o4; unsigned short o5; unsigned short o6; unsigned short o7;
    asm volatile("atom.global.add.noftz.v8.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3), "=h"(o4), "=h"(o5), "=h"(o6), "=h"(o7) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; r[4] = o4; r[5] = o5; r[6] = o6; r[7] = o7; }
  for (int i = 0; i < 8; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v2.f16x2 [%0], {%1,%2};" :: "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
  { unsigned o0; unsigned o1;
    asm volatile("atom.global.add.noftz.v2.f16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(o0), "=r"(o1) : "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v4.f16x2 [%0], {%1,%2,%3,%4};" :: "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
  { unsigned o0; unsigned o1; unsigned o2; unsigned o3;
    asm volatile("atom.global.add.noftz.v4.f16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3) : "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1,%2};" :: "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
  { unsigned o0; unsigned o1;
    asm volatile("atom.global.add.noftz.v2.bf16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(o0), "=r"(o1) : "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1,%2,%3,%4};" :: "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
  { unsigned o0; unsigned o1; unsigned o2; unsigned o3;
    asm volatile("atom.global.add.noftz.v4.bf16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3) : "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v2.f16 [%0], {%1,%2};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
  { unsigned short o0; unsigned short o1;
    asm volatile("atom.global.min.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};" : "=h"(o0), "=h"(o1) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v4.f16 [%0], {%1,%2,%3,%4};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3;
    asm volatile("atom.global.min.noftz.v4.f16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v8.f16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3; unsigned short o4; unsigned short o5; unsigned short o6; unsigned short o7;
    asm volatile("atom.global.min.noftz.v8.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3), "=h"(o4), "=h"(o5), "=h"(o6), "=h"(o7) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; r[4] = o4; r[5] = o5; r[6] = o6; r[7] = o7; }
  for (int i = 0; i < 8; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v2.bf16 [%0], {%1,%2};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
  { unsigned short o0; unsigned short o1;
    asm volatile("atom.global.min.noftz.v2.bf16 {%0,%1}, [%2], {%3,%4};" : "=h"(o0), "=h"(o1) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v4.bf16 [%0], {%1,%2,%3,%4};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3;
    asm volatile("atom.global.min.noftz.v4.bf16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v8.bf16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3; unsigned short o4; unsigned short o5; unsigned short o6; unsigned short o7;
    asm volatile("atom.global.min.noftz.v8.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3), "=h"(o4), "=h"(o5), "=h"(o6), "=h"(o7) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; r[4] = o4; r[5] = o5; r[6] = o6; r[7] = o7; }
  for (int i = 0; i < 8; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v2.f16x2 [%0], {%1,%2};" :: "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
  { unsigned o0; unsigned o1;
    asm volatile("atom.global.min.noftz.v2.f16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(o0), "=r"(o1) : "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v4.f16x2 [%0], {%1,%2,%3,%4};" :: "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
  { unsigned o0; unsigned o1; unsigned o2; unsigned o3;
    asm volatile("atom.global.min.noftz.v4.f16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3) : "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v2.bf16x2 [%0], {%1,%2};" :: "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
  { unsigned o0; unsigned o1;
    asm volatile("atom.global.min.noftz.v2.bf16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(o0), "=r"(o1) : "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.min.noftz.v4.bf16x2 [%0], {%1,%2,%3,%4};" :: "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
  { unsigned o0; unsigned o1; unsigned o2; unsigned o3;
    asm volatile("atom.global.min.noftz.v4.bf16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3) : "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v2.f16 [%0], {%1,%2};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
  { unsigned short o0; unsigned short o1;
    asm volatile("atom.global.max.noftz.v2.f16 {%0,%1}, [%2], {%3,%4};" : "=h"(o0), "=h"(o1) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v4.f16 [%0], {%1,%2,%3,%4};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3;
    asm volatile("atom.global.max.noftz.v4.f16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v8.f16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3; unsigned short o4; unsigned short o5; unsigned short o6; unsigned short o7;
    asm volatile("atom.global.max.noftz.v8.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3), "=h"(o4), "=h"(o5), "=h"(o6), "=h"(o7) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; r[4] = o4; r[5] = o5; r[6] = o6; r[7] = o7; }
  for (int i = 0; i < 8; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v2.bf16 [%0], {%1,%2};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
  { unsigned short o0; unsigned short o1;
    asm volatile("atom.global.max.noftz.v2.bf16 {%0,%1}, [%2], {%3,%4};" : "=h"(o0), "=h"(o1) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v4.bf16 [%0], {%1,%2,%3,%4};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3;
    asm volatile("atom.global.max.noftz.v4.bf16 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v8.bf16 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
  { unsigned short o0; unsigned short o1; unsigned short o2; unsigned short o3; unsigned short o4; unsigned short o5; unsigned short o6; unsigned short o7;
    asm volatile("atom.global.max.noftz.v8.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], {%9,%10,%11,%12,%13,%14,%15,%16};" : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3), "=h"(o4), "=h"(o5), "=h"(o6), "=h"(o7) : "l"(a), "h"((unsigned short)v[0]), "h"((unsigned short)v[1]), "h"((unsigned short)v[2]), "h"((unsigned short)v[3]), "h"((unsigned short)v[4]), "h"((unsigned short)v[5]), "h"((unsigned short)v[6]), "h"((unsigned short)v[7]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; r[4] = o4; r[5] = o5; r[6] = o6; r[7] = o7; }
  for (int i = 0; i < 8; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v2.f16x2 [%0], {%1,%2};" :: "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
  { unsigned o0; unsigned o1;
    asm volatile("atom.global.max.noftz.v2.f16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(o0), "=r"(o1) : "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v4.f16x2 [%0], {%1,%2,%3,%4};" :: "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
  { unsigned o0; unsigned o1; unsigned o2; unsigned o3;
    asm volatile("atom.global.max.noftz.v4.f16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3) : "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v2.bf16x2 [%0], {%1,%2};" :: "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
  { unsigned o0; unsigned o1;
    asm volatile("atom.global.max.noftz.v2.bf16x2 {%0,%1}, [%2], {%3,%4};" : "=r"(o0), "=r"(o1) : "l"(a), "r"(v[0]), "r"(v[1]) : "memory");
    r[0] = o0; r[1] = o1; }
  for (int i = 0; i < 2; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
  asm volatile("red.global.max.noftz.v4.bf16x2 [%0], {%1,%2,%3,%4};" :: "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
  { unsigned o0; unsigned o1; unsigned o2; unsigned o3;
    asm volatile("atom.global.max.noftz.v4.bf16x2 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};" : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3) : "l"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]) : "memory");
    r[0] = o0; r[1] = o1; r[2] = o2; r[3] = o3; }
  for (int i = 0; i < 4; ++i) mem[4096 + threadIdx.x * 8 + i] = r[i];
}
