// Hopper's DPX intrinsics (VIMNMX, VIADDMNMX and VIMNMX3 with their .RELU and packed forms): the decoder
// is checked against nvdisasm on these.
#include <cstdio>
__global__ void k(unsigned* o, const unsigned* in) {
  unsigned a = in[0], b = in[1], c = in[2]; bool p, ph, pl;
  o[0] = __viaddmax_s16x2(a, b, c);
  o[1] = __viaddmax_s16x2_relu(a, b, c);
  o[2] = __viaddmin_u16x2(a, b, c);
  o[3] = __vimax3_s16x2_relu(a, b, c);
  o[4] = __vimin3_u16x2(a, b, c);
  o[5] = __viaddmax_s32(a, b, c);
  o[6] = __viaddmax_s32_relu(a, b, c);
  o[7] = __viaddmin_s32(a, b, c);
  o[8] = __viaddmin_u32(a, b, c);
  o[9] = __vimax3_s32_relu(a, b, c);
  o[10] = __vimax_s32_relu(a, b);
  o[11] = __vimin_s16x2_relu(a, b);
  o[13] = __vibmax_s32(a, b, &p); o[14] = p;
  o[15] = __vibmax_s16x2(a, b, &ph, &pl); o[16] = ph * 2 + pl;
  o[17] = __viaddmax_u16x2(a, b, c);
  o[18] = __viaddmin_s16x2(a, b, c);
  o[19] = __viaddmin_s16x2_relu(a, b, c);
  o[20] = __vimin3_u32(a, b, c);
  o[21] = __vimax3_s16x2(a, b, c);
  o[22] = min(a, b);
  o[23] = max((int)a, (int)b);
  o[24] = __vimax3_u32(a, b, c);
  o[25] = __vimin3_s32(a, b, c);
  o[26] = __viaddmin_s32_relu(a, b, c);
  unsigned t;
  asm("add.u16x2 %0, %1, %2;" : "=r"(t) : "r"(a), "r"(b));      // VIADD.16x2, the packed add on its own
  o[27] = t;
  asm("add.s16x2 %0, %1, %2;" : "=r"(t) : "r"(b), "r"(c));
  o[28] = t;
}
