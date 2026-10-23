| build (SASS+PTX of) | native A10G, embedded SASS | native A10G, CUDA_FORCE_PTX_JIT=1 | first native failure |
|---|---|---|---|
| sm_80 | 16/16 PASS | 16/16 PASS |  |
| sm_86 | 16/16 PASS | 16/16 PASS |  |
| sm_89 | 1/16 PASS | 1/16 PASS | FAIL int_alu |
| sm_90 | 1/16 PASS | 1/16 PASS | FAIL int_alu |
| sm_120 | 1/16 PASS | 1/16 PASS | FAIL int_alu |

| program build | simulated profile | sim engine | sim PASS | native A10G (JIT of PTX / SASS) can run it? | workloads where sim fails |
|---|---|---|---|---|---|
| sm_80 | nvidia/a100 | PTX x1, SASS x15 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/a10g | PTX x1, SASS x15 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_89 | nvidia/l4 | PTX x1, SASS x15 | 16/16 | SASS 1/16, PTX-JIT 1/16 | - |
| sm_89 | nvidia/l40s | PTX x1, SASS x15 | 16/16 | SASS 1/16, PTX-JIT 1/16 | - |
| sm_90 | nvidia/h100 | PTX x1, SASS x15 | 16/16 | SASS 1/16, PTX-JIT 1/16 | - |
| sm_120 | nvidia/rtx5090 | PTX x1, SASS x15 | 16/16 | SASS 1/16, PTX-JIT 1/16 | - |
| sm_86 | nvidia/l4 | PTX x1, SASS x15 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/l40s | PTX x1, SASS x15 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/h100 | PTX x16 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/rtx5090 | PTX x16 | 16/16 | SASS 16/16, PTX-JIT 16/16 | - |
| sm_86 | nvidia/a100 | PTX x16 | 1/16 | SASS 16/16, PTX-JIT 16/16 | int_alu:FAIL int_alu; f32_fma:FAIL f32_fma; f64_fma:FAIL f64_fma; f64_subnormal:FAIL f64_subnormal; wmma:FAIL wmma; mma_ptx:FAIL mma_ptx; mem:FAIL mem; shared:FAIL shared; atomics:FAIL atomics; textur |
| sm_86 | nvidia/t4 | PTX x13 | 0/13 | SASS 16/16, PTX-JIT 16/16 | int_alu:FAIL int_alu; f32_fma:FAIL f32_fma; f64_fma:FAIL f64_fma; f64_subnormal:FAIL f64_subnormal; wmma:FAIL wmma; mma_ptx:FAIL mma_ptx; mem:FAIL mem; shared:FAIL shared; atomics:FAIL atomics; textur |
