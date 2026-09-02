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
