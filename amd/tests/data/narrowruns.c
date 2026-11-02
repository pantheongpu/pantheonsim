// Global loads and stores of a byte or a short whose lanes take any addresses the test hands them -- consecutive, strided,
// repeated, overlapping, misaligned, out of range -- with some lanes switched off. The test runs each pattern with the
// interpreter's whole-run path on and off and compares memory, errors and counts (test_gcn_narrowruns.cpp).
typedef unsigned long u64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef __attribute__((address_space(1))) char gchar;
typedef __attribute__((address_space(1))) const char gcchar;

// dst[doff[t]] = src[soff[t]] for each lane t whose on[t] is set; byte offsets, so any alignment. LD is the type read (a byte or a short,
// signed or not), ST the type written: a narrow one copies the element, an int writes the widened value.
#define COPY(NAME, LD, ST)                                                                                     \
  __attribute__((amdgpu_kernel)) void NAME(gcchar* src, gchar* dst, const u64* soff, const u64* doff,       \
                                           const unsigned* on) {                                               \
    int t = __builtin_amdgcn_workitem_id_x();                                                                  \
    if (on[t]) {                                                                                               \
      LD v = *(const volatile __attribute__((address_space(1))) LD*)(src + soff[t]);                           \
      *(volatile __attribute__((address_space(1))) ST*)(dst + doff[t]) = (ST)v;                                \
    }                                                                                                          \
  }
COPY(copy8, u8, u8)
COPY(copy16, u16, u16)
COPY(load8u, u8, int)
COPY(load8s, signed char, int)
COPY(load16u, u16, int)
COPY(load16s, short, int)

// The same with 32-bit offsets, which the compiler adds to a scalar base in the instruction itself.
#define COPYS(NAME, LD, ST)                                                                                    \
  __attribute__((amdgpu_kernel)) void NAME(gcchar* src, gchar* dst, const unsigned* soff,                   \
                                           const unsigned* doff, const unsigned* on) {                         \
    int t = __builtin_amdgcn_workitem_id_x();                                                                  \
    if (on[t]) {                                                                                               \
      LD v = *(const volatile __attribute__((address_space(1))) LD*)(src + soff[t]);                           \
      *(volatile __attribute__((address_space(1))) ST*)(dst + doff[t]) = (ST)v;                                \
    }                                                                                                          \
  }
COPYS(copy8s, u8, u8)
COPYS(copy16s, u16, u16)
COPYS(load8ss, signed char, int)
COPYS(load16ss, short, int)
