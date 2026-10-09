// Formatted surface stores (sust.p): one to four 32-bit values converted to the surface's format.
#include <cuda_runtime.h>
extern "C" __global__ void sust_p_1d(cudaSurfaceObject_t s, const unsigned* in) {
  asm volatile("sust.p.1d.v4.b32.trap [%0, {%1}], {%2,%3,%4,%5};" :: "l"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]), "r"(in[4]));
  asm volatile("sust.p.1d.v2.b32.clamp [%0, {%1}], {%2,%3};" :: "l"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]));
  asm volatile("sust.p.1d.b32.zero [%0, {%1}], {%2};" :: "l"(s), "r"(in[0]), "r"(in[1]));
}
extern "C" __global__ void sust_p_2d(cudaSurfaceObject_t s, const unsigned* in) {
  asm volatile("sust.p.2d.v4.b32.trap [%0, {%1,%2}], {%3,%4,%5,%6};" :: "l"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]), "r"(in[4]), "r"(in[5]));
  asm volatile("sust.p.2d.v2.b32.clamp [%0, {%1,%2}], {%3,%4};" :: "l"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]));
  asm volatile("sust.p.2d.b32.zero [%0, {%1,%2}], {%3};" :: "l"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]));
}
extern "C" __global__ void sust_p_3d(cudaSurfaceObject_t s, const unsigned* in) {
  asm volatile("sust.p.3d.v4.b32.trap [%0, {%1,%2,%3,%3}], {%4,%5,%6,%7};" :: "l"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]), "r"(in[4]), "r"(in[5]), "r"(in[6]));
  asm volatile("sust.p.3d.v2.b32.clamp [%0, {%1,%2,%3,%3}], {%4,%5};" :: "l"(s), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3]), "r"(in[4]));
}
