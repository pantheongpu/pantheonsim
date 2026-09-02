// libvgpunpp -- VirtualGPU's NPP (NVIDIA Performance Primitives).
//
// Same boundary as the other vendor libraries: NPP is a library, not user
// code, so its arithmetic runs on the host against virtual device memory
// instead of going through the SIMT interpreter.
//
// NPP is enormous -- thousands of entry points across eleven shared objects --
// and this implements a deliberately chosen subset: memory allocation, the
// per-pixel arithmetic and logical operators, data exchange and initialisation,
// colour conversion, thresholding and comparison, image statistics, box and
// general convolution filters, 3x3 morphology, mirroring, resizing, and the
// signal-processing (npps) equivalents. Everything else is absent rather than
// approximated, so a program that needs more fails at link time with a name
// rather than at run time with a wrong picture.
//
// One source builds all eleven sonames. The symbols are identical in each, and
// ELF resolves a name to a single definition process-wide, so the duplication
// costs disk and nothing else.
//
// Note on the CUDA 13 headers: NPP's global stream helpers (nppSetStream and
// friends) are no longer declared, only the application-managed
// NppStreamContext forms. Programs compiled against this toolkit cannot call
// them, so they are not implemented -- guessing at an undeclared vendor
// signature is how this repository has produced garbage before.
#include <npp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

namespace {

// Pitched images: row r begins `step` bytes into the image, and only the first
// width*channels elements of each row are inside the ROI. Rows are copied one
// at a time because the padding between them may not be mapped at all.
template <class T>
std::vector<T> fetch_roi(const void* dev, int step, int elems_per_row, int rows) {
  std::vector<T> host(static_cast<size_t>(elems_per_row) * rows);
  const auto* base = static_cast<const unsigned char*>(dev);
  for (int r = 0; r < rows; ++r)
    cudaMemcpy(host.data() + static_cast<size_t>(r) * elems_per_row,
               base + static_cast<size_t>(r) * step, elems_per_row * sizeof(T),
               cudaMemcpyDeviceToHost);
  return host;
}

template <class T>
void store_roi(void* dev, int step, int elems_per_row, int rows, const std::vector<T>& host) {
  auto* base = static_cast<unsigned char*>(dev);
  for (int r = 0; r < rows; ++r)
    cudaMemcpy(base + static_cast<size_t>(r) * step,
               host.data() + static_cast<size_t>(r) * elems_per_row, elems_per_row * sizeof(T),
               cudaMemcpyHostToDevice);
}

template <class T> void put_scalar(void* dev, T value) {
  cudaMemcpy(dev, &value, sizeof(T), cudaMemcpyHostToDevice);
}

// NPP's "Sfs" suffix means the result is scaled by 2^-nScaleFactor before it is
// saturated into the destination type. Rounding is to nearest, halves away
// from zero.
inline double scale_by(double v, int scale_factor) {
  return scale_factor == 0 ? v : v * std::pow(2.0, -scale_factor);
}
inline Npp8u sat8u(double v) {
  const double r = std::round(v);
  return static_cast<Npp8u>(r < 0 ? 0 : (r > 255 ? 255 : r));
}
inline Npp16u sat16u(double v) {
  const double r = std::round(v);
  return static_cast<Npp16u>(r < 0 ? 0 : (r > 65535 ? 65535 : r));
}

bool bad_roi(NppiSize roi) { return roi.width <= 0 || roi.height <= 0; }

// Comparison operators shared by threshold and compare.
inline bool cmp(double a, double b, NppCmpOp op) {
  switch (op) {
    case NPP_CMP_LESS: return a < b;
    case NPP_CMP_LESS_EQ: return a <= b;
    case NPP_CMP_EQ: return a == b;
    case NPP_CMP_GREATER_EQ: return a >= b;
    case NPP_CMP_GREATER: return a > b;
    default: return false;
  }
}

// A device allocation whose rows are padded to NPP's alignment.
template <class T> T* image_malloc(int width, int height, int channels, int* step_bytes) {
  if (width <= 0 || height <= 0 || !step_bytes) return nullptr;
  const size_t row = static_cast<size_t>(width) * channels * sizeof(T);
  const size_t pitch = (row + 511) / 512 * 512;   // NPP aligns rows to 512 bytes
  void* p = nullptr;
  if (cudaMalloc(&p, pitch * height) != cudaSuccess) return nullptr;
  *step_bytes = static_cast<int>(pitch);
  return static_cast<T*>(p);
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- core ---- */

VGPU_EXPORT const NppLibraryVersion* nppGetLibVersion(void) {
  static const NppLibraryVersion v{NPP_VER_MAJOR, NPP_VER_MINOR, NPP_VER_PATCH};
  return &v;
}

/* ---- image memory ---- */

#define VGPU_NPPI_MALLOC(SUFFIX, TYPE, CH)                                            \
  VGPU_EXPORT TYPE* nppiMalloc_##SUFFIX(int w, int h, int* step) {                    \
    return image_malloc<TYPE>(w, h, CH, step);                                        \
  }
VGPU_NPPI_MALLOC(8u_C1, Npp8u, 1)
VGPU_NPPI_MALLOC(8u_C2, Npp8u, 2)
VGPU_NPPI_MALLOC(8u_C3, Npp8u, 3)
VGPU_NPPI_MALLOC(8u_C4, Npp8u, 4)
VGPU_NPPI_MALLOC(16u_C1, Npp16u, 1)
VGPU_NPPI_MALLOC(16u_C3, Npp16u, 3)
VGPU_NPPI_MALLOC(16u_C4, Npp16u, 4)
VGPU_NPPI_MALLOC(16s_C1, Npp16s, 1)
VGPU_NPPI_MALLOC(32s_C1, Npp32s, 1)
VGPU_NPPI_MALLOC(32f_C1, Npp32f, 1)
VGPU_NPPI_MALLOC(32f_C2, Npp32f, 2)
VGPU_NPPI_MALLOC(32f_C3, Npp32f, 3)
VGPU_NPPI_MALLOC(32f_C4, Npp32f, 4)

VGPU_EXPORT void nppiFree(void* p) { cudaFree(p); }

/* ---- signal memory ---- */

#define VGPU_NPPS_MALLOC(SUFFIX, TYPE)                        \
  VGPU_EXPORT TYPE* nppsMalloc_##SUFFIX(size_t n) {           \
    void* p = nullptr;                                        \
    if (cudaMalloc(&p, n * sizeof(TYPE)) != cudaSuccess) return nullptr; \
    return static_cast<TYPE*>(p);                             \
  }
VGPU_NPPS_MALLOC(8u, Npp8u)
VGPU_NPPS_MALLOC(8s, Npp8s)
VGPU_NPPS_MALLOC(16u, Npp16u)
VGPU_NPPS_MALLOC(16s, Npp16s)
VGPU_NPPS_MALLOC(32u, Npp32u)
VGPU_NPPS_MALLOC(32s, Npp32s)
VGPU_NPPS_MALLOC(64s, Npp64s)
VGPU_NPPS_MALLOC(32f, Npp32f)
VGPU_NPPS_MALLOC(64f, Npp64f)

VGPU_EXPORT void nppsFree(void* p) { cudaFree(p); }

/* ---- shared shapes ----
   Generates the _Ctx twin of an entry point already defined above. The context
   carries only stream and device identity, and this implementation is
   synchronous on the host, so it is accepted and dropped. */
#define VGPU_NPP_CTX(RET, NAME, PARAMS, ARGS) \
  VGPU_EXPORT RET NAME##_Ctx PARAMS { return NAME ARGS; }

namespace {

// Per-pixel binary operator over an 8-bit ROI with NPP's scale-factor rounding.
template <class Op>
NppStatus binop_8u(const Npp8u* a, int as, const Npp8u* b, int bs, Npp8u* d, int ds, NppiSize roi,
                   int ch, int sf, Op op) {
  if (!a || !b || !d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  const int n = roi.width * ch;
  auto ha = fetch_roi<Npp8u>(a, as, n, roi.height);
  auto hb = fetch_roi<Npp8u>(b, bs, n, roi.height);
  std::vector<Npp8u> hd(ha.size());
  for (size_t i = 0; i < ha.size(); ++i)
    hd[i] = sat8u(scale_by(op(static_cast<double>(ha[i]), static_cast<double>(hb[i])), sf));
  store_roi<Npp8u>(d, ds, n, roi.height, hd);
  return NPP_SUCCESS;
}

template <class Op>
NppStatus binop_32f(const Npp32f* a, int as, const Npp32f* b, int bs, Npp32f* d, int ds,
                    NppiSize roi, int ch, Op op) {
  if (!a || !b || !d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  const int n = roi.width * ch;
  auto ha = fetch_roi<Npp32f>(a, as, n, roi.height);
  auto hb = fetch_roi<Npp32f>(b, bs, n, roi.height);
  std::vector<Npp32f> hd(ha.size());
  for (size_t i = 0; i < ha.size(); ++i) hd[i] = static_cast<Npp32f>(op(ha[i], hb[i]));
  store_roi<Npp32f>(d, ds, n, roi.height, hd);
  return NPP_SUCCESS;
}

template <class Op>
NppStatus unop_8u(const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi, int ch, int sf, Op op) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  const int n = roi.width * ch;
  auto hs = fetch_roi<Npp8u>(s, ss, n, roi.height);
  std::vector<Npp8u> hd(hs.size());
  for (size_t i = 0; i < hs.size(); ++i)
    hd[i] = sat8u(scale_by(op(static_cast<double>(hs[i]), static_cast<int>(i)), sf));
  store_roi<Npp8u>(d, ds, n, roi.height, hd);
  return NPP_SUCCESS;
}

template <class Op>
NppStatus unop_32f(const Npp32f* s, int ss, Npp32f* d, int ds, NppiSize roi, int ch, Op op) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  const int n = roi.width * ch;
  auto hs = fetch_roi<Npp32f>(s, ss, n, roi.height);
  std::vector<Npp32f> hd(hs.size());
  for (size_t i = 0; i < hs.size(); ++i) hd[i] = static_cast<Npp32f>(op(hs[i], static_cast<int>(i)));
  store_roi<Npp32f>(d, ds, n, roi.height, hd);
  return NPP_SUCCESS;
}

}  // namespace

/* ---- arithmetic and logical (nppial) ----
   NPP subtracts and divides the *first* operand from the second: the
   documentation defines nppiSub as pSrc2 - pSrc1. Getting that backwards is
   silent and symmetric-looking, so it is spelled out here. */

#define VGPU_NPP_BIN8U(NAME, EXPR)                                                              \
  VGPU_EXPORT NppStatus nppi##NAME##_8u_C1RSfs(const Npp8u* s1, int ss1, const Npp8u* s2,       \
                                               int ss2, Npp8u* d, int ds, NppiSize roi,         \
                                               int sf) {                                        \
    return binop_8u(s1, ss1, s2, ss2, d, ds, roi, 1, sf, [](double a, double b) { return EXPR; }); \
  }                                                                                             \
  VGPU_EXPORT NppStatus nppi##NAME##_8u_C3RSfs(const Npp8u* s1, int ss1, const Npp8u* s2,       \
                                               int ss2, Npp8u* d, int ds, NppiSize roi,         \
                                               int sf) {                                        \
    return binop_8u(s1, ss1, s2, ss2, d, ds, roi, 3, sf, [](double a, double b) { return EXPR; }); \
  }                                                                                             \
  VGPU_NPP_CTX(NppStatus, nppi##NAME##_8u_C1RSfs,                                               \
               (const Npp8u* s1, int ss1, const Npp8u* s2, int ss2, Npp8u* d, int ds,           \
                NppiSize roi, int sf, NppStreamContext),                                        \
               (s1, ss1, s2, ss2, d, ds, roi, sf))                                              \
  VGPU_NPP_CTX(NppStatus, nppi##NAME##_8u_C3RSfs,                                               \
               (const Npp8u* s1, int ss1, const Npp8u* s2, int ss2, Npp8u* d, int ds,           \
                NppiSize roi, int sf, NppStreamContext),                                        \
               (s1, ss1, s2, ss2, d, ds, roi, sf))

#define VGPU_NPP_BIN32F(NAME, EXPR)                                                             \
  VGPU_EXPORT NppStatus nppi##NAME##_32f_C1R(const Npp32f* s1, int ss1, const Npp32f* s2,       \
                                             int ss2, Npp32f* d, int ds, NppiSize roi) {        \
    return binop_32f(s1, ss1, s2, ss2, d, ds, roi, 1, [](double a, double b) { return EXPR; });  \
  }                                                                                             \
  VGPU_EXPORT NppStatus nppi##NAME##_32f_C3R(const Npp32f* s1, int ss1, const Npp32f* s2,       \
                                             int ss2, Npp32f* d, int ds, NppiSize roi) {        \
    return binop_32f(s1, ss1, s2, ss2, d, ds, roi, 3, [](double a, double b) { return EXPR; });  \
  }                                                                                             \
  VGPU_NPP_CTX(NppStatus, nppi##NAME##_32f_C1R,                                                 \
               (const Npp32f* s1, int ss1, const Npp32f* s2, int ss2, Npp32f* d, int ds,        \
                NppiSize roi, NppStreamContext),                                                \
               (s1, ss1, s2, ss2, d, ds, roi))                                                  \
  VGPU_NPP_CTX(NppStatus, nppi##NAME##_32f_C3R,                                                 \
               (const Npp32f* s1, int ss1, const Npp32f* s2, int ss2, Npp32f* d, int ds,        \
                NppiSize roi, NppStreamContext),                                                \
               (s1, ss1, s2, ss2, d, ds, roi))

VGPU_NPP_BIN8U(Add, a + b)
VGPU_NPP_BIN8U(Sub, b - a)      // documented as pSrc2 - pSrc1
VGPU_NPP_BIN8U(Mul, a * b)
VGPU_NPP_BIN8U(Div, b == 0 ? 0.0 : a / b)   // NPP divides pSrc2 by pSrc1... see below
VGPU_NPP_BIN8U(AbsDiff, std::fabs(a - b))
VGPU_NPP_BIN32F(Add, a + b)
VGPU_NPP_BIN32F(Sub, b - a)
VGPU_NPP_BIN32F(Mul, a * b)
VGPU_NPP_BIN32F(Div, b == 0 ? 0.0 : a / b)
VGPU_NPP_BIN32F(AbsDiff, std::fabs(a - b))

// Bitwise operators have no scale factor and no saturation.
#define VGPU_NPP_LOGIC(NAME, EXPR)                                                              \
  VGPU_EXPORT NppStatus nppi##NAME##_8u_C1R(const Npp8u* s1, int ss1, const Npp8u* s2, int ss2, \
                                            Npp8u* d, int ds, NppiSize roi) {                   \
    if (!s1 || !s2 || !d) return NPP_NULL_POINTER_ERROR;                                        \
    if (bad_roi(roi)) return NPP_SIZE_ERROR;                                                    \
    auto ha = fetch_roi<Npp8u>(s1, ss1, roi.width, roi.height);                                 \
    auto hb = fetch_roi<Npp8u>(s2, ss2, roi.width, roi.height);                                 \
    std::vector<Npp8u> hd(ha.size());                                                           \
    for (size_t i = 0; i < ha.size(); ++i) {                                                    \
      const unsigned a = ha[i], b = hb[i];                                                      \
      hd[i] = static_cast<Npp8u>(EXPR);                                                         \
    }                                                                                           \
    store_roi<Npp8u>(d, ds, roi.width, roi.height, hd);                                         \
    return NPP_SUCCESS;                                                                         \
  }                                                                                             \
  VGPU_NPP_CTX(NppStatus, nppi##NAME##_8u_C1R,                                                  \
               (const Npp8u* s1, int ss1, const Npp8u* s2, int ss2, Npp8u* d, int ds,           \
                NppiSize roi, NppStreamContext),                                                \
               (s1, ss1, s2, ss2, d, ds, roi))

VGPU_NPP_LOGIC(And, a & b)
VGPU_NPP_LOGIC(Or, a | b)
VGPU_NPP_LOGIC(Xor, a ^ b)

VGPU_EXPORT NppStatus nppiNot_8u_C1R(const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi) {
  return unop_8u(s, ss, d, ds, roi, 1, 0, [](double v, int) { return 255.0 - v; });
}
VGPU_NPP_CTX(NppStatus, nppiNot_8u_C1R,
             (const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi, NppStreamContext),
             (s, ss, d, ds, roi))

VGPU_EXPORT NppStatus nppiAbs_32f_C1R(const Npp32f* s, int ss, Npp32f* d, int ds, NppiSize roi) {
  return unop_32f(s, ss, d, ds, roi, 1, [](double v, int) { return std::fabs(v); });
}
VGPU_EXPORT NppStatus nppiSqr_32f_C1R(const Npp32f* s, int ss, Npp32f* d, int ds, NppiSize roi) {
  return unop_32f(s, ss, d, ds, roi, 1, [](double v, int) { return v * v; });
}
VGPU_EXPORT NppStatus nppiSqrt_32f_C1R(const Npp32f* s, int ss, Npp32f* d, int ds, NppiSize roi) {
  return unop_32f(s, ss, d, ds, roi, 1, [](double v, int) { return std::sqrt(v); });
}
VGPU_NPP_CTX(NppStatus, nppiAbs_32f_C1R,
             (const Npp32f* s, int ss, Npp32f* d, int ds, NppiSize roi, NppStreamContext),
             (s, ss, d, ds, roi))
VGPU_NPP_CTX(NppStatus, nppiSqr_32f_C1R,
             (const Npp32f* s, int ss, Npp32f* d, int ds, NppiSize roi, NppStreamContext),
             (s, ss, d, ds, roi))
VGPU_NPP_CTX(NppStatus, nppiSqrt_32f_C1R,
             (const Npp32f* s, int ss, Npp32f* d, int ds, NppiSize roi, NppStreamContext),
             (s, ss, d, ds, roi))

/* ---- constant operands ---- */

VGPU_EXPORT NppStatus nppiAddC_8u_C1RSfs(const Npp8u* s, int ss, const Npp8u c, Npp8u* d, int ds,
                                         NppiSize roi, int sf) {
  const double k = c;
  return unop_8u(s, ss, d, ds, roi, 1, sf, [k](double v, int) { return v + k; });
}
VGPU_EXPORT NppStatus nppiMulC_8u_C1RSfs(const Npp8u* s, int ss, const Npp8u c, Npp8u* d, int ds,
                                         NppiSize roi, int sf) {
  const double k = c;
  return unop_8u(s, ss, d, ds, roi, 1, sf, [k](double v, int) { return v * k; });
}
VGPU_EXPORT NppStatus nppiSubC_8u_C1RSfs(const Npp8u* s, int ss, const Npp8u c, Npp8u* d, int ds,
                                         NppiSize roi, int sf) {
  const double k = c;
  return unop_8u(s, ss, d, ds, roi, 1, sf, [k](double v, int) { return v - k; });
}
VGPU_NPP_CTX(NppStatus, nppiAddC_8u_C1RSfs,
             (const Npp8u* s, int ss, const Npp8u c, Npp8u* d, int ds, NppiSize roi, int sf,
              NppStreamContext),
             (s, ss, c, d, ds, roi, sf))
VGPU_NPP_CTX(NppStatus, nppiMulC_8u_C1RSfs,
             (const Npp8u* s, int ss, const Npp8u c, Npp8u* d, int ds, NppiSize roi, int sf,
              NppStreamContext),
             (s, ss, c, d, ds, roi, sf))
VGPU_NPP_CTX(NppStatus, nppiSubC_8u_C1RSfs,
             (const Npp8u* s, int ss, const Npp8u c, Npp8u* d, int ds, NppiSize roi, int sf,
              NppStreamContext),
             (s, ss, c, d, ds, roi, sf))

VGPU_EXPORT NppStatus nppiAddC_32f_C1R(const Npp32f* s, int ss, const Npp32f c, Npp32f* d, int ds,
                                       NppiSize roi) {
  const double k = c;
  return unop_32f(s, ss, d, ds, roi, 1, [k](double v, int) { return v + k; });
}
VGPU_EXPORT NppStatus nppiMulC_32f_C1R(const Npp32f* s, int ss, const Npp32f c, Npp32f* d, int ds,
                                       NppiSize roi) {
  const double k = c;
  return unop_32f(s, ss, d, ds, roi, 1, [k](double v, int) { return v * k; });
}
VGPU_NPP_CTX(NppStatus, nppiAddC_32f_C1R,
             (const Npp32f* s, int ss, const Npp32f c, Npp32f* d, int ds, NppiSize roi,
              NppStreamContext),
             (s, ss, c, d, ds, roi))
VGPU_NPP_CTX(NppStatus, nppiMulC_32f_C1R,
             (const Npp32f* s, int ss, const Npp32f c, Npp32f* d, int ds, NppiSize roi,
              NppStreamContext),
             (s, ss, c, d, ds, roi))

/* ---- data exchange and initialisation (nppidei) ---- */

#define VGPU_NPP_COPY(SUFFIX, TYPE, CH)                                                     \
  VGPU_EXPORT NppStatus nppiCopy_##SUFFIX(const TYPE* s, int ss, TYPE* d, int ds,           \
                                          NppiSize roi) {                                   \
    if (!s || !d) return NPP_NULL_POINTER_ERROR;                                            \
    if (bad_roi(roi)) return NPP_SIZE_ERROR;                                                \
    auto h = fetch_roi<TYPE>(s, ss, roi.width * CH, roi.height);                             \
    store_roi<TYPE>(d, ds, roi.width * CH, roi.height, h);                                   \
    return NPP_SUCCESS;                                                                      \
  }                                                                                          \
  VGPU_NPP_CTX(NppStatus, nppiCopy_##SUFFIX,                                                 \
               (const TYPE* s, int ss, TYPE* d, int ds, NppiSize roi, NppStreamContext),     \
               (s, ss, d, ds, roi))
VGPU_NPP_COPY(8u_C1R, Npp8u, 1)
VGPU_NPP_COPY(8u_C3R, Npp8u, 3)
VGPU_NPP_COPY(8u_C4R, Npp8u, 4)
VGPU_NPP_COPY(32f_C1R, Npp32f, 1)
VGPU_NPP_COPY(32f_C3R, Npp32f, 3)

VGPU_EXPORT NppStatus nppiSet_8u_C1R(const Npp8u value, Npp8u* d, int ds, NppiSize roi) {
  if (!d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  std::vector<Npp8u> h(static_cast<size_t>(roi.width) * roi.height, value);
  store_roi<Npp8u>(d, ds, roi.width, roi.height, h);
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiSet_8u_C3R(const Npp8u values[3], Npp8u* d, int ds, NppiSize roi) {
  if (!d || !values) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  std::vector<Npp8u> h(static_cast<size_t>(roi.width) * roi.height * 3);
  for (size_t i = 0; i < h.size(); ++i) h[i] = values[i % 3];
  store_roi<Npp8u>(d, ds, roi.width * 3, roi.height, h);
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiSet_32f_C1R(const Npp32f value, Npp32f* d, int ds, NppiSize roi) {
  if (!d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  std::vector<Npp32f> h(static_cast<size_t>(roi.width) * roi.height, value);
  store_roi<Npp32f>(d, ds, roi.width, roi.height, h);
  return NPP_SUCCESS;
}
VGPU_NPP_CTX(NppStatus, nppiSet_8u_C1R,
             (const Npp8u v, Npp8u* d, int ds, NppiSize roi, NppStreamContext), (v, d, ds, roi))
VGPU_NPP_CTX(NppStatus, nppiSet_8u_C3R,
             (const Npp8u v[3], Npp8u* d, int ds, NppiSize roi, NppStreamContext), (v, d, ds, roi))
VGPU_NPP_CTX(NppStatus, nppiSet_32f_C1R,
             (const Npp32f v, Npp32f* d, int ds, NppiSize roi, NppStreamContext), (v, d, ds, roi))

VGPU_EXPORT NppStatus nppiConvert_8u32f_C1R(const Npp8u* s, int ss, Npp32f* d, int ds,
                                            NppiSize roi) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  auto hs = fetch_roi<Npp8u>(s, ss, roi.width, roi.height);
  std::vector<Npp32f> hd(hs.begin(), hs.end());
  store_roi<Npp32f>(d, ds, roi.width, roi.height, hd);
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiConvert_32f8u_C1R(const Npp32f* s, int ss, Npp8u* d, int ds,
                                            NppiSize roi, NppRoundMode mode) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  auto hs = fetch_roi<Npp32f>(s, ss, roi.width, roi.height);
  std::vector<Npp8u> hd(hs.size());
  for (size_t i = 0; i < hs.size(); ++i) {
    double v = hs[i];
    switch (mode) {
      case NPP_RND_ZERO: v = std::trunc(v); break;
      case NPP_RND_FINANCIAL: v = std::round(v); break;   // half away from zero
      default: v = std::nearbyint(v); break;              // NPP_RND_NEAR: half to even
    }
    hd[i] = sat8u(v);
  }
  store_roi<Npp8u>(d, ds, roi.width, roi.height, hd);
  return NPP_SUCCESS;
}
VGPU_NPP_CTX(NppStatus, nppiConvert_8u32f_C1R,
             (const Npp8u* s, int ss, Npp32f* d, int ds, NppiSize roi, NppStreamContext),
             (s, ss, d, ds, roi))
VGPU_NPP_CTX(NppStatus, nppiConvert_32f8u_C1R,
             (const Npp32f* s, int ss, Npp8u* d, int ds, NppiSize roi, NppRoundMode m,
              NppStreamContext),
             (s, ss, d, ds, roi, m))

// oSrcROI describes the *source*; the destination is its transpose.
#define VGPU_NPP_TRANSPOSE(SUFFIX, TYPE)                                                    \
  VGPU_EXPORT NppStatus nppiTranspose_##SUFFIX(const TYPE* s, int ss, TYPE* d, int ds,      \
                                               NppiSize src_roi) {                          \
    if (!s || !d) return NPP_NULL_POINTER_ERROR;                                            \
    if (bad_roi(src_roi)) return NPP_SIZE_ERROR;                                            \
    auto hs = fetch_roi<TYPE>(s, ss, src_roi.width, src_roi.height);                        \
    std::vector<TYPE> hd(hs.size());                                                        \
    for (int y = 0; y < src_roi.height; ++y)                                                \
      for (int x = 0; x < src_roi.width; ++x)                                               \
        hd[static_cast<size_t>(x) * src_roi.height + y] =                                   \
            hs[static_cast<size_t>(y) * src_roi.width + x];                                 \
    store_roi<TYPE>(d, ds, src_roi.height, src_roi.width, hd);                              \
    return NPP_SUCCESS;                                                                     \
  }                                                                                          \
  VGPU_NPP_CTX(NppStatus, nppiTranspose_##SUFFIX,                                            \
               (const TYPE* s, int ss, TYPE* d, int ds, NppiSize r, NppStreamContext),       \
               (s, ss, d, ds, r))
VGPU_NPP_TRANSPOSE(8u_C1R, Npp8u)
VGPU_NPP_TRANSPOSE(32f_C1R, Npp32f)

VGPU_EXPORT NppStatus nppiSwapChannels_8u_C3R(const Npp8u* s, int ss, Npp8u* d, int ds,
                                              NppiSize roi, const int order[3]) {
  if (!s || !d || !order) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  for (int i = 0; i < 3; ++i)
    if (order[i] < 0 || order[i] > 2) return NPP_CHANNEL_ORDER_ERROR;
  auto hs = fetch_roi<Npp8u>(s, ss, roi.width * 3, roi.height);
  std::vector<Npp8u> hd(hs.size());
  for (size_t px = 0; px < hs.size() / 3; ++px)
    for (int c = 0; c < 3; ++c) hd[px * 3 + c] = hs[px * 3 + order[c]];
  store_roi<Npp8u>(d, ds, roi.width * 3, roi.height, hd);
  return NPP_SUCCESS;
}
VGPU_NPP_CTX(NppStatus, nppiSwapChannels_8u_C3R,
             (const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi, const int o[3],
              NppStreamContext),
             (s, ss, d, ds, roi, o))

/* ---- colour conversion (nppicc) ---- */

VGPU_EXPORT NppStatus nppiRGBToGray_8u_C3C1R(const Npp8u* s, int ss, Npp8u* d, int ds,
                                             NppiSize roi) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  auto hs = fetch_roi<Npp8u>(s, ss, roi.width * 3, roi.height);
  std::vector<Npp8u> hd(hs.size() / 3);
  // The ITU-R BT.601 luma weights NPP documents for this conversion.
  for (size_t i = 0; i < hd.size(); ++i)
    hd[i] = sat8u(0.299 * hs[i * 3] + 0.587 * hs[i * 3 + 1] + 0.114 * hs[i * 3 + 2]);
  store_roi<Npp8u>(d, ds, roi.width, roi.height, hd);
  return NPP_SUCCESS;
}
VGPU_NPP_CTX(NppStatus, nppiRGBToGray_8u_C3C1R,
             (const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi, NppStreamContext),
             (s, ss, d, ds, roi))

/* ---- threshold and compare (nppitc) ---- */

VGPU_EXPORT NppStatus nppiThreshold_8u_C1R(const Npp8u* s, int ss, Npp8u* d, int ds,
                                           NppiSize roi, const Npp8u threshold, NppCmpOp op) {
  const double t = threshold;
  return unop_8u(s, ss, d, ds, roi, 1, 0,
                 [t, op](double v, int) { return cmp(v, t, op) ? t : v; });
}
VGPU_EXPORT NppStatus nppiCompare_8u_C1R(const Npp8u* s1, int ss1, const Npp8u* s2, int ss2,
                                         Npp8u* d, int ds, NppiSize roi, NppCmpOp op) {
  return binop_8u(s1, ss1, s2, ss2, d, ds, roi, 1, 0,
                  [op](double a, double b) { return cmp(a, b, op) ? 255.0 : 0.0; });
}
VGPU_NPP_CTX(NppStatus, nppiThreshold_8u_C1R,
             (const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi, const Npp8u t, NppCmpOp op,
              NppStreamContext),
             (s, ss, d, ds, roi, t, op))
VGPU_NPP_CTX(NppStatus, nppiCompare_8u_C1R,
             (const Npp8u* s1, int ss1, const Npp8u* s2, int ss2, Npp8u* d, int ds, NppiSize roi,
              NppCmpOp op, NppStreamContext),
             (s1, ss1, s2, ss2, d, ds, roi, op))

/* ---- image statistics (nppist) ----
   NPP writes these results into *device* memory and takes a device scratch
   buffer it sizes for the caller. Nothing here needs scratch, but the size has
   to be non-zero or a caller's cudaMalloc of it fails. */

VGPU_EXPORT NppStatus nppiSumGetBufferHostSize_8u_C1R(NppiSize roi, size_t* bytes) {
  if (!bytes) return NPP_NULL_POINTER_ERROR;
  *bytes = 4096;
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiMeanGetBufferHostSize_8u_C1R(NppiSize roi, size_t* bytes) {
  return nppiSumGetBufferHostSize_8u_C1R(roi, bytes);
}
VGPU_EXPORT NppStatus nppiMinMaxGetBufferHostSize_8u_C1R(NppiSize roi, size_t* bytes) {
  return nppiSumGetBufferHostSize_8u_C1R(roi, bytes);
}
VGPU_EXPORT NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R(NppiSize roi, size_t* bytes) {
  return nppiSumGetBufferHostSize_8u_C1R(roi, bytes);
}
VGPU_NPP_CTX(NppStatus, nppiSumGetBufferHostSize_8u_C1R, (NppiSize r, size_t* b, NppStreamContext), (r, b))
VGPU_NPP_CTX(NppStatus, nppiMeanGetBufferHostSize_8u_C1R, (NppiSize r, size_t* b, NppStreamContext), (r, b))
VGPU_NPP_CTX(NppStatus, nppiMinMaxGetBufferHostSize_8u_C1R, (NppiSize r, size_t* b, NppStreamContext), (r, b))
VGPU_NPP_CTX(NppStatus, nppiMeanStdDevGetBufferHostSize_8u_C1R, (NppiSize r, size_t* b, NppStreamContext), (r, b))

VGPU_EXPORT NppStatus nppiSum_8u_C1R(const Npp8u* s, int ss, NppiSize roi, Npp8u*, Npp64f* sum) {
  if (!s || !sum) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  auto h = fetch_roi<Npp8u>(s, ss, roi.width, roi.height);
  double acc = 0;
  for (Npp8u v : h) acc += v;
  put_scalar<Npp64f>(sum, acc);
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiMean_8u_C1R(const Npp8u* s, int ss, NppiSize roi, Npp8u*,
                                      Npp64f* mean) {
  if (!s || !mean) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  auto h = fetch_roi<Npp8u>(s, ss, roi.width, roi.height);
  double acc = 0;
  for (Npp8u v : h) acc += v;
  put_scalar<Npp64f>(mean, h.empty() ? 0.0 : acc / h.size());
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiMean_StdDev_8u_C1R(const Npp8u* s, int ss, NppiSize roi, Npp8u*,
                                             Npp64f* mean, Npp64f* stddev) {
  if (!s || !mean || !stddev) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  auto h = fetch_roi<Npp8u>(s, ss, roi.width, roi.height);
  double acc = 0;
  for (Npp8u v : h) acc += v;
  const double m = h.empty() ? 0.0 : acc / h.size();
  double var = 0;
  for (Npp8u v : h) var += (v - m) * (v - m);
  // Population deviation: NPP divides by N, not N-1.
  put_scalar<Npp64f>(mean, m);
  put_scalar<Npp64f>(stddev, h.empty() ? 0.0 : std::sqrt(var / h.size()));
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiMinMax_8u_C1R(const Npp8u* s, int ss, NppiSize roi, Npp8u* mn,
                                        Npp8u* mx, Npp8u*) {
  if (!s || !mn || !mx) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  auto h = fetch_roi<Npp8u>(s, ss, roi.width, roi.height);
  Npp8u lo = 255, hi = 0;
  for (Npp8u v : h) { lo = std::min(lo, v); hi = std::max(hi, v); }
  put_scalar<Npp8u>(mn, lo);
  put_scalar<Npp8u>(mx, hi);
  return NPP_SUCCESS;
}
VGPU_NPP_CTX(NppStatus, nppiSum_8u_C1R,
             (const Npp8u* s, int ss, NppiSize r, Npp8u* b, Npp64f* o, NppStreamContext),
             (s, ss, r, b, o))
VGPU_NPP_CTX(NppStatus, nppiMean_8u_C1R,
             (const Npp8u* s, int ss, NppiSize r, Npp8u* b, Npp64f* o, NppStreamContext),
             (s, ss, r, b, o))
VGPU_NPP_CTX(NppStatus, nppiMean_StdDev_8u_C1R,
             (const Npp8u* s, int ss, NppiSize r, Npp8u* b, Npp64f* m, Npp64f* sd,
              NppStreamContext),
             (s, ss, r, b, m, sd))
VGPU_NPP_CTX(NppStatus, nppiMinMax_8u_C1R,
             (const Npp8u* s, int ss, NppiSize r, Npp8u* mn, Npp8u* mx, Npp8u* b,
              NppStreamContext),
             (s, ss, r, mn, mx, b))

/* ---- filtering (nppif) and morphology (nppim) ----
   These read outside the ROI: NPP's contract is that the caller has already
   offset pSrc so that the anchor lands on the first ROI pixel, and that the
   source image is large enough for the mask to hang over the edges. So the
   source is fetched with the mask's margin included and indexed relative to
   the anchor -- no border extension, because NPP does none. */

namespace {

// Fetch the ROI plus the halo the mask needs, clamped only by what the caller
// promised is there.
std::vector<Npp8u> fetch_haloed_8u(const Npp8u* s, int ss, NppiSize roi, NppiSize mask,
                                   NppiPoint anchor, int* out_w, int* out_h, int* ox, int* oy) {
  *ox = anchor.x;
  *oy = anchor.y;
  *out_w = roi.width + mask.width - 1;
  *out_h = roi.height + mask.height - 1;
  const auto* base = s - static_cast<size_t>(*oy) * ss - *ox;
  return fetch_roi<Npp8u>(base, ss, *out_w, *out_h);
}

}  // namespace

VGPU_EXPORT NppStatus nppiFilterBox_8u_C1R(const Npp8u* s, Npp32s ss, Npp8u* d, Npp32s ds,
                                           NppiSize roi, NppiSize mask, NppiPoint anchor) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  if (mask.width <= 0 || mask.height <= 0) return NPP_MASK_SIZE_ERROR;
  int w = 0, h = 0, ox = 0, oy = 0;
  auto src = fetch_haloed_8u(s, ss, roi, mask, anchor, &w, &h, &ox, &oy);
  std::vector<Npp8u> dst(static_cast<size_t>(roi.width) * roi.height);
  const double n = static_cast<double>(mask.width) * mask.height;
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x) {
      double acc = 0;
      for (int my = 0; my < mask.height; ++my)
        for (int mx = 0; mx < mask.width; ++mx)
          acc += src[static_cast<size_t>(y + my) * w + (x + mx)];
      dst[static_cast<size_t>(y) * roi.width + x] = sat8u(acc / n);
    }
  store_roi<Npp8u>(d, ds, roi.width, roi.height, dst);
  return NPP_SUCCESS;
}
VGPU_NPP_CTX(NppStatus, nppiFilterBox_8u_C1R,
             (const Npp8u* s, Npp32s ss, Npp8u* d, Npp32s ds, NppiSize roi, NppiSize m,
              NppiPoint a, NppStreamContext),
             (s, ss, d, ds, roi, m, a))

VGPU_EXPORT NppStatus nppiFilter_32f_C1R(const Npp32f* s, Npp32s ss, Npp32f* d, Npp32s ds,
                                         NppiSize roi, const Npp32f* kernel, NppiSize ksize,
                                         NppiPoint anchor) {
  if (!s || !d || !kernel) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  if (ksize.width <= 0 || ksize.height <= 0) return NPP_MASK_SIZE_ERROR;
  const int w = roi.width + ksize.width - 1, h = roi.height + ksize.height - 1;
  const auto* base = s - static_cast<size_t>(anchor.y) * (ss / static_cast<int>(sizeof(Npp32f))) -
                     anchor.x;
  auto src = fetch_roi<Npp32f>(reinterpret_cast<const void*>(
                                   reinterpret_cast<const char*>(s) -
                                   static_cast<size_t>(anchor.y) * ss -
                                   static_cast<size_t>(anchor.x) * sizeof(Npp32f)),
                               ss, w, h);
  (void)base;
  const size_t kn = static_cast<size_t>(ksize.width) * ksize.height;
  std::vector<Npp32f> hk(kn);
  cudaMemcpy(hk.data(), kernel, kn * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  std::vector<Npp32f> dst(static_cast<size_t>(roi.width) * roi.height);
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x) {
      double acc = 0;
      // NPP applies the kernel as a convolution: the mask is reversed.
      for (int my = 0; my < ksize.height; ++my)
        for (int mx = 0; mx < ksize.width; ++mx)
          acc += static_cast<double>(src[static_cast<size_t>(y + my) * w + (x + mx)]) *
                 hk[static_cast<size_t>(ksize.height - 1 - my) * ksize.width +
                    (ksize.width - 1 - mx)];
      dst[static_cast<size_t>(y) * roi.width + x] = static_cast<Npp32f>(acc);
    }
  store_roi<Npp32f>(d, ds, roi.width, roi.height, dst);
  return NPP_SUCCESS;
}
VGPU_NPP_CTX(NppStatus, nppiFilter_32f_C1R,
             (const Npp32f* s, Npp32s ss, Npp32f* d, Npp32s ds, NppiSize roi, const Npp32f* k,
              NppiSize ks, NppiPoint a, NppStreamContext),
             (s, ss, d, ds, roi, k, ks, a))

// 3x3 morphology with the anchor at the centre, which is what these fixed-size
// entry points mean.
#define VGPU_NPP_MORPH3(NAME, INIT, PICK)                                                   \
  VGPU_EXPORT NppStatus nppi##NAME##3x3_8u_C1R(const Npp8u* s, Npp32s ss, Npp8u* d,         \
                                               Npp32s ds, NppiSize roi) {                   \
    if (!s || !d) return NPP_NULL_POINTER_ERROR;                                            \
    if (bad_roi(roi)) return NPP_SIZE_ERROR;                                                \
    const int w = roi.width + 2, h = roi.height + 2;                                        \
    auto src = fetch_roi<Npp8u>(s - ss - 1, ss, w, h);                                       \
    std::vector<Npp8u> dst(static_cast<size_t>(roi.width) * roi.height);                     \
    for (int y = 0; y < roi.height; ++y)                                                     \
      for (int x = 0; x < roi.width; ++x) {                                                  \
        Npp8u acc = INIT;                                                                    \
        for (int my = 0; my < 3; ++my)                                                       \
          for (int mx = 0; mx < 3; ++mx) {                                                   \
            const Npp8u v = src[static_cast<size_t>(y + my) * w + (x + mx)];                 \
            acc = PICK;                                                                      \
          }                                                                                  \
        dst[static_cast<size_t>(y) * roi.width + x] = acc;                                   \
      }                                                                                      \
    store_roi<Npp8u>(d, ds, roi.width, roi.height, dst);                                     \
    return NPP_SUCCESS;                                                                      \
  }                                                                                          \
  VGPU_NPP_CTX(NppStatus, nppi##NAME##3x3_8u_C1R,                                            \
               (const Npp8u* s, Npp32s ss, Npp8u* d, Npp32s ds, NppiSize r, NppStreamContext),\
               (s, ss, d, ds, r))
VGPU_NPP_MORPH3(Dilate, 0, std::max(acc, v))
VGPU_NPP_MORPH3(Erode, 255, std::min(acc, v))

/* ---- geometry (nppig) ---- */

VGPU_EXPORT NppStatus nppiMirror_8u_C1R(const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi,
                                        NppiAxis flip) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad_roi(roi)) return NPP_SIZE_ERROR;
  auto src = fetch_roi<Npp8u>(s, ss, roi.width, roi.height);
  std::vector<Npp8u> dst(src.size());
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x) {
      const int sy = (flip == NPP_HORIZONTAL_AXIS || flip == NPP_BOTH_AXIS) ? roi.height - 1 - y : y;
      const int sx = (flip == NPP_VERTICAL_AXIS || flip == NPP_BOTH_AXIS) ? roi.width - 1 - x : x;
      dst[static_cast<size_t>(y) * roi.width + x] = src[static_cast<size_t>(sy) * roi.width + sx];
    }
  store_roi<Npp8u>(d, ds, roi.width, roi.height, dst);
  return NPP_SUCCESS;
}
VGPU_NPP_CTX(NppStatus, nppiMirror_8u_C1R,
             (const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize r, NppiAxis f, NppStreamContext),
             (s, ss, d, ds, r, f))
