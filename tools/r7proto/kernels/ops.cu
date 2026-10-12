// Throwaway: kernels that put every operation the prototype supports through values that break
// arithmetic. Each thread reads x, y, z and writes one result per expression.
#include <cstdint>

#define OUT(k, v) out[(size_t)(k) * n + i] = (v)

extern "C" __global__ void ops_int(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = a[i], y = b[i], z = c[i];
  OUT(0, x + y + z);
  OUT(1, x - y + z);
  OUT(2, x - y - z);
  OUT(3, x * y + z);
  OUT(4, x * y - z);
  OUT(5, (uint32_t)((int32_t)x * (int32_t)y) + z);
  OUT(6, (x & y) | z);
  OUT(7, x ^ y ^ z);
  OUT(8, (~x & y) ^ z);
  OUT(9, (x | y) & ~z);
  OUT(10, (x & y) ^ (~z));
  OUT(11, ~(x ^ y));
  OUT(12, x + 0x7fffffffu);
  OUT(13, x - 1u);
  OUT(14, -x);
  OUT(15, ~x + y);
}

extern "C" __global__ void ops_wide(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = a[i], y = b[i], z = c[i];
  const uint64_t u = (uint64_t)x * y + z;
  const uint64_t u2 = (uint64_t)x * y + ((uint64_t)z << 32 | x);
  const int64_t s = (int64_t)(int32_t)x * (int32_t)y + (int64_t)(int32_t)z;
  const int64_t s2 = (int64_t)(int32_t)x * (int32_t)y + (((int64_t)(int32_t)z << 32) | y);
  OUT(0, (uint32_t)u);
  OUT(1, (uint32_t)(u >> 32));
  OUT(2, (uint32_t)u2);
  OUT(3, (uint32_t)(u2 >> 32));
  OUT(4, (uint32_t)s);
  OUT(5, (uint32_t)((uint64_t)s >> 32));
  OUT(6, (uint32_t)s2);
  OUT(7, (uint32_t)((uint64_t)s2 >> 32));
}

extern "C" __global__ void ops_shift(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = a[i], y = b[i], z = c[i];
  OUT(0, x << (y & 31));
  OUT(1, x >> (y & 31));
  OUT(2, (uint32_t)((int32_t)x >> (y & 31)));
  OUT(3, __funnelshift_l(x, z, y & 31));
  OUT(4, __funnelshift_r(x, z, y & 31));
  OUT(5, __funnelshift_lc(x, z, y));
  OUT(6, __funnelshift_rc(x, z, y));
  OUT(7, (x << 7) | (x >> 25));
  OUT(8, (x >> 3) + (y << 4));
  OUT(9, x << y);
  OUT(10, x >> y);
  OUT(11, (uint32_t)((int32_t)x >> y));
  OUT(12, (uint32_t)(((uint64_t)x << 32 | y) >> (z & 63)));
  OUT(13, (uint32_t)(((int64_t)(int32_t)x << 17) >> (z & 15)));
}

extern "C" __global__ void ops_float(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = __uint_as_float(a[i]), y = __uint_as_float(b[i]), z = __uint_as_float(c[i]);
  OUT(0, __float_as_uint(x + y));
  OUT(1, __float_as_uint(x * y));
  OUT(2, __float_as_uint(__fmaf_rn(x, y, z)));
  OUT(3, __float_as_uint(__fmaf_rn(-x, y, z)));
  OUT(4, __float_as_uint(__fmaf_rn(x, -y, -z)));
  OUT(5, __float_as_uint(__fmaf_rn(fabsf(x), y, fabsf(z))));
  OUT(6, __float_as_uint(__fadd_rn(-x, y)));
  OUT(7, __float_as_uint(__fmul_rn(fabsf(x), -fabsf(y))));
  OUT(8, __float_as_uint(__fmaf_rn(x, 0.5f, y)));
  OUT(9, __float_as_uint(x - y));
  OUT(10, __float_as_uint(__fadd_rn(x, 1.0f)));
}

extern "C" __global__ void ops_cvt(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = a[i], y = b[i], z = c[i];
  OUT(0, __float_as_uint(__int2float_rn((int32_t)x)));
  OUT(1, __float_as_uint(__uint2float_rn(x)));
  OUT(2, __float_as_uint(__int2float_rn((int32_t)y) + __uint2float_rn(z)));
}

// Predicates: guarded stores (ISETP with every comparison, signed and unsigned, and the bool ops).
extern "C" __global__ void ops_pred(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = a[i], y = b[i], z = c[i];
  uint32_t* o = out + i;
  const size_t s = n;
  if (x < y) o[0 * s] = x;
  if ((int32_t)x < (int32_t)y) o[1 * s] = y;
  if (x <= y) o[2 * s] = z;
  if ((int32_t)x <= (int32_t)y) o[3 * s] = x + z;
  if (x == y) o[4 * s] = 1;
  if (x != y) o[5 * s] = 2;
  if (x > z) o[6 * s] = 3;
  if ((int32_t)x > (int32_t)z) o[7 * s] = 4;
  if (x >= z) o[8 * s] = 5;
  if ((int32_t)x >= (int32_t)z) o[9 * s] = 6;
  if (x < y && y < z) o[10 * s] = 7;
  if (x < y || y < z) o[11 * s] = 8;
  if ((x ^ y) & 1) o[12 * s] = x ^ y;
}
