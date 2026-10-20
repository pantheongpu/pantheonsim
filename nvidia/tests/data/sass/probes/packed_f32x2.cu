// Blackwell's packed fp32 (FFMA2, FMUL2, FADD2): the end-to-end test's program is the probe -- every operand form, swizzle and
// sign modifier ptxas emits for add/mul/fma.{rn,rz,rm,rp}{,.ftz}.f32x2. sm_100 and later only (older targets do not compile it
// and are skipped; sm_120 lowers it to scalar instructions).
#include "../../../e2e/packed_f32x2.cu"
