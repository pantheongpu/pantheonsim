// Global loads and stores whose lanes take any addresses the test hands them -- consecutive, strided, repeated,
// overlapping, misaligned, out of range -- at one, two and four words a lane, with some lanes switched off. The
// test runs each pattern with the interpreter's whole-run path on and off and compares memory, errors and counts.
typedef unsigned long u64;
typedef int v2 __attribute__((ext_vector_type(2)));
typedef int v4 __attribute__((ext_vector_type(4)));
// Global (not generic) pointers, so the accesses are global_load/store and not flat.
typedef __attribute__((address_space(1))) char gchar;
typedef __attribute__((address_space(1))) const char gcchar;

// dst[doff[t]] = src[soff[t]] for each lane t whose on[t] is set; byte offsets, so any alignment.
#define COPY(NAME, T)                                                                                          \
  __attribute__((amdgpu_kernel)) void NAME(gcchar* src, gchar* dst, const u64* soff, const u64* doff,       \
                                           const unsigned* on) {                                               \
    int t = __builtin_amdgcn_workitem_id_x();                                                                  \
    if (on[t]) {                                                                                               \
      T v = *(const volatile __attribute__((address_space(1))) T*)(src + soff[t]);                                                               \
      *(volatile __attribute__((address_space(1))) T*)(dst + doff[t]) = v;                                                                       \
    }                                                                                                          \
  }
COPY(copy32, int)
COPY(copy64, u64)
COPY(copy64v, v2)
COPY(copy128, v4)

// The same with 32-bit offsets, which the compiler adds to a scalar base in the instruction itself.
#define COPYS(NAME, T)                                                                                         \
  __attribute__((amdgpu_kernel)) void NAME(gcchar* src, gchar* dst, const unsigned* soff,                   \
                                           const unsigned* doff, const unsigned* on) {                         \
    int t = __builtin_amdgcn_workitem_id_x();                                                                  \
    if (on[t]) {                                                                                               \
      T v = *(const volatile __attribute__((address_space(1))) T*)(src + soff[t]);                                                               \
      *(volatile __attribute__((address_space(1))) T*)(dst + doff[t]) = v;                                                                       \
    }                                                                                                          \
  }
COPYS(copy32s, int)
COPYS(copy64s, u64)
COPYS(copy128s, v4)
