// Decoder probe: every IDP and VABSDIFF4 form ptxas emits for sm_86 (the
// repo's own tests reach only a few). Compiled into the SASS corpus.
#define P(i, s) asm volatile(s " %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); o[i] = r;
__global__ void k(unsigned* o, unsigned a, unsigned b, unsigned c) {
  unsigned r;
  P(0, "dp2a.lo.s32.s32") P(1, "dp2a.hi.s32.s32") P(2, "dp2a.lo.u32.u32") P(3, "dp2a.hi.u32.u32")
  P(4, "dp2a.lo.s32.u32") P(5, "dp2a.hi.u32.s32") P(6, "dp4a.s32.u32") P(7, "dp4a.u32.s32")
  P(8, "vabsdiff4.u32.u32.u32") P(9, "vabsdiff4.u32.u32.u32.add") P(10, "vabsdiff4.u32.u32.u32.sat")
  P(11, "vabsdiff4.s32.s32.s32")  P(13, "vabsdiff4.s32.s32.s32.add")
  P(14, "vabsdiff4.u32.s32.s32.add") P(15, "vabsdiff2.u32.u32.u32.add")
}
