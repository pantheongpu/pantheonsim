// Arrays, textures and surfaces on the Radeon GPUs, which have texture units:
// HIP's texture and surface functions compiled for each RDNA generation
// (images.gfx1030, images.gfx1100, images.gfx1201), each result against the
// host. Every check prints "<what>: <n> of <m> wrong".
//
//   - 2D float arrays: point samples in normalized and unnormalized
//     coordinates, bilinear filtering, and the wrap, clamp, mirror and
//     border address modes;
//   - 8-bit arrays read as normalized floats and as integers, and an
//     explicit level of detail with no mipmaps;
//   - linear memory through tex1Dfetch (and the driver API's
//     hipTexObjectCreate), pitched memory through tex2D;
//   - 3D, 2D layered and 1D layered arrays, with hipMemcpy3D into them and
//     hipMemcpy3DBatchAsync between them;
//   - gathers, and surfaces written and read;
//   - the copies to and from arrays, and what the API answers about them.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Texture references, bound to linear memory and to an array. At file
// scope, as a program declares them: in an anonymous namespace the device
// compiler would take the never-written variable as a constant.
texture<float, 1, hipReadModeElementType> g_linear_ref;
texture<float, 2, hipReadModeElementType> g_array_ref;
__global__ void fetch_reference(float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex1Dfetch(g_linear_ref, i);
}
__global__ void sample_reference(float* out, int w, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < w * h) out[i] = tex2D(g_array_ref, i % w + 0.5f, i / w + 0.5f);
}

namespace {

void report(const char* what, int wrong, int of) { std::printf("%s: %d of %d wrong\n", what, wrong, of); }

constexpr int W = 8, H = 4;
float f2d(int x, int y) { return static_cast<float>(x + 10 * y) + 0.25f; }
int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

__global__ void sample2d(hipTextureObject_t t, const float2* at, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2D<float>(t, at[i].x, at[i].y);
}
__global__ void sample2d_uchar(hipTextureObject_t t, const float2* at, uchar4* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2D<uchar4>(t, at[i].x, at[i].y);
}
__global__ void sample2d_norm(hipTextureObject_t t, const float2* at, float4* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2D<float4>(t, at[i].x, at[i].y);
}
__global__ void sample2d_lod(hipTextureObject_t t, const float2* at, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2DLod<float>(t, at[i].x, at[i].y, 2.0f);
}
// HIP's tex2Dgather numbers the components its own way: 1 is red, 2 green,
// 3 blue, and anything else (0 included) alpha.
__global__ void gather2d(hipTextureObject_t t, const float2* at, float4* out, int n, int comp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex2Dgather<float4>(t, at[i].x, at[i].y, comp);
}
__global__ void fetch1d(hipTextureObject_t t, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex1Dfetch<float>(t, i);
}
__global__ void fetch1d_int2(hipTextureObject_t t, int2* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tex1Dfetch<int2>(t, i);
}
__global__ void sample3d(hipTextureObject_t t, float* out, int n, int d) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * n * d) out[i] = tex3D<float>(t, i % n + 0.5f, (i / n) % n + 0.5f, i / (n * n) + 0.5f);
}
__global__ void sample2d_layered(hipTextureObject_t t, float* out, int n, int layers) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * n * layers) out[i] = tex2DLayered<float>(t, i % n + 0.5f, (i / n) % n + 0.5f, i / (n * n));
}
__global__ void sample1d_layered(hipTextureObject_t t, float* out, int n, int layers) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * layers) out[i] = tex1DLayered<float>(t, i % n + 0.5f, i / n);
}
__global__ void surface_write(hipSurfaceObject_t s) {
  const int x = threadIdx.x, y = blockIdx.x;
  surf2Dwrite(static_cast<float>(x * 100 + y), s, x * 4, y);
}
__global__ void surface_read(hipSurfaceObject_t s, float* out) {
  const int x = threadIdx.x, y = blockIdx.x;
  float v;
  surf2Dread(&v, s, x * 4, y);
  out[y * blockDim.x + x] = v;
}

// A cube's face addressed as a layer: face 0 of a 2D surface is the surface.
__global__ void surface_cube_face(hipSurfaceObject_t s, float* out) {
  const int x = threadIdx.x, y = blockIdx.x;
  surfCubemapwrite(static_cast<float>(x - y), s, x * 4, y, 0);
  float v;
  surfCubemapread(&v, s, x * 4, y, 0);
  out[y * blockDim.x + x] = v;
}

template <typename T>
T* device_copy(const std::vector<T>& v) {
  T* p = nullptr;
  if (hipMalloc(&p, v.size() * sizeof(T)) != hipSuccess) return nullptr;
  (void)hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice);
  return p;
}
template <typename T>
std::vector<T> host_copy(const T* p, size_t n) {
  std::vector<T> v(n);
  (void)hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost);
  return v;
}

hipTextureDesc tex_desc(hipTextureAddressMode mode, hipTextureFilterMode filter, int normalized,
                        hipTextureReadMode read = hipReadModeElementType) {
  hipTextureDesc t;
  std::memset(&t, 0, sizeof t);
  t.addressMode[0] = t.addressMode[1] = t.addressMode[2] = mode;
  t.filterMode = filter;
  t.readMode = read;
  t.normalizedCoords = normalized;
  return t;
}

}  // namespace

int main() {
  int images = 0;
  CHECK(hipDeviceGetAttribute(&images, hipDeviceAttributeImageSupport, 0));
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  report("image support and texture limits", (images != 1) + (p.maxTexture2D[0] != 16384) +
                                                 (p.maxTexture3D[2] != 8192) + (p.textureAlignment != 256),
         4);

  // ---- A 2D float array, sampled every way ----
  std::vector<float> texels(W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) texels[y * W + x] = f2d(x, y);
  const hipChannelFormatDesc fdesc = hipCreateChannelDesc<float>();
  hipArray_t arr = nullptr;
  CHECK(hipMallocArray(&arr, &fdesc, W, H));
  CHECK(hipMemcpy2DToArray(arr, 0, 0, texels.data(), W * sizeof(float), W * sizeof(float), H,
                           hipMemcpyHostToDevice));
  {
    std::vector<float> back(W * H, -1);
    CHECK(hipMemcpy2DFromArray(back.data(), W * sizeof(float), arr, 0, 0, W * sizeof(float), H,
                               hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int i = 0; i < W * H; ++i) wrong += back[i] != texels[i];
    // And from device memory into another array, as a device-to-device copy
    // without compute units (hipMemcpyDeviceToDeviceNoCU).
    float* d_rows = nullptr;
    CHECK(hipMalloc(&d_rows, W * H * sizeof(float)));
    CHECK(hipMemcpy(d_rows, texels.data(), W * H * sizeof(float), hipMemcpyHostToDevice));
    hipArray_t twin = nullptr;
    CHECK(hipMallocArray(&twin, &fdesc, W, H));
    CHECK(hipMemcpy2DToArray(twin, 0, 0, d_rows, W * sizeof(float), W * sizeof(float), H,
                             hipMemcpyDeviceToDeviceNoCU));
    std::fill(back.begin(), back.end(), -1.0f);
    CHECK(hipMemcpy2DFromArray(back.data(), W * sizeof(float), twin, 0, 0, W * sizeof(float), H,
                               hipMemcpyDeviceToHost));
    for (int i = 0; i < W * H; ++i) wrong += back[i] != texels[i];
    CHECK(hipFreeArray(twin));
    CHECK(hipFree(d_rows));
    report("a 2D array copied in and out, from the host and from device memory", wrong, 2 * W * H);
  }
  hipResourceDesc res;
  std::memset(&res, 0, sizeof res);
  res.resType = hipResourceTypeArray;
  res.res.array.array = arr;

  const int n = W * H;
  std::vector<float2> at(n);
  float2* d_at = nullptr;
  float* d_out = nullptr;
  CHECK(hipMalloc(&d_at, 64 * sizeof(float2)));
  CHECK(hipMalloc(&d_out, 64 * sizeof(float4)));
  const auto run = [&](hipTextureObject_t t, const std::vector<float2>& coords) {
    (void)hipMemcpy(d_at, coords.data(), coords.size() * sizeof(float2), hipMemcpyHostToDevice);
    sample2d<<<1, 64>>>(t, d_at, d_out, static_cast<int>(coords.size()));
    return host_copy(d_out, coords.size());
  };

  // Point samples at each texel's center, normalized and not.
  {
    hipTextureObject_t tn = 0, tu = 0;
    hipTextureDesc d = tex_desc(hipAddressModeClamp, hipFilterModePoint, 1);
    CHECK(hipCreateTextureObject(&tn, &res, &d, nullptr));
    d.normalizedCoords = 0;
    CHECK(hipCreateTextureObject(&tu, &res, &d, nullptr));
    for (int i = 0; i < n; ++i) at[i] = make_float2((i % W + 0.5f) / W, (i / W + 0.5f) / H);
    std::vector<float> r = run(tn, at);
    int wrong = 0;
    for (int i = 0; i < n; ++i) wrong += r[i] != f2d(i % W, i / W);
    for (int i = 0; i < n; ++i) at[i] = make_float2(i % W + 0.5f, i / W + 0.5f);
    r = run(tu, at);
    for (int i = 0; i < n; ++i) wrong += r[i] != f2d(i % W, i / W);
    report("point samples at texel centers", wrong, 2 * n);
    // An explicit level of detail with no mipmaps reads the only level.
    for (int i = 0; i < n; ++i) at[i] = make_float2((i % W + 0.5f) / W, (i / W + 0.5f) / H);
    (void)hipMemcpy(d_at, at.data(), n * sizeof(float2), hipMemcpyHostToDevice);
    sample2d_lod<<<1, 64>>>(tn, d_at, d_out, n);
    r = host_copy(d_out, n);
    wrong = 0;
    for (int i = 0; i < n; ++i) wrong += r[i] != f2d(i % W, i / W);
    report("a level of detail with no mipmaps", wrong, n);
    CHECK(hipDestroyTextureObject(tn));
    CHECK(hipDestroyTextureObject(tu));
  }
  // Bilinear: a quarter texel right of each center weighs its right
  // neighbour a quarter (clamped at the edge).
  {
    hipTextureObject_t t = 0;
    hipTextureDesc d = tex_desc(hipAddressModeClamp, hipFilterModeLinear, 0);
    CHECK(hipCreateTextureObject(&t, &res, &d, nullptr));
    for (int i = 0; i < n; ++i) at[i] = make_float2(i % W + 0.75f, i / W + 0.5f);
    const std::vector<float> r = run(t, at);
    int wrong = 0;
    for (int i = 0; i < n; ++i) {
      const int x = i % W, y = i / W;
      const float want = 0.75f * f2d(x, y) + 0.25f * f2d(clampi(x + 1, 0, W - 1), y);
      wrong += std::fabs(r[i] - want) > 1e-4f;
    }
    report("bilinear samples", wrong, n);
    CHECK(hipDestroyTextureObject(t));
  }
  // The address modes, past each edge.
  {
    struct Mode {
      hipTextureAddressMode mode;
      const char* name;
    } modes[] = {{hipAddressModeWrap, "wrap"}, {hipAddressModeClamp, "clamp"}, {hipAddressModeMirror, "mirror"},
                 {hipAddressModeBorder, "border"}};
    int wrong = 0;
    for (const Mode& m : modes) {
      hipTextureObject_t t = 0;
      hipTextureDesc d = tex_desc(m.mode, hipFilterModePoint, 1);
      CHECK(hipCreateTextureObject(&t, &res, &d, nullptr));
      // x one period to the right; y a texel above the top.
      for (int i = 0; i < W; ++i) at[i] = make_float2(1.0f + (i + 0.5f) / W, -0.5f / H);
      const std::vector<float> r = run(t, std::vector<float2>(at.begin(), at.begin() + W));
      for (int x = 0; x < W; ++x) {
        float want = 0;
        switch (m.mode) {
          case hipAddressModeWrap: want = f2d(x, H - 1); break;
          case hipAddressModeClamp: want = f2d(W - 1, 0); break;
          case hipAddressModeMirror: want = f2d(W - 1 - x, 0); break;
          default: want = 0; break;
        }
        if (r[x] != want) {
          ++wrong;
          std::printf("  %s x=%d: %g, not %g\n", m.name, x, r[x], want);
        }
      }
      CHECK(hipDestroyTextureObject(t));
    }
    report("wrap, clamp, mirror and border", wrong, 4 * W);
  }
  // Gathers: at each texel's bottom-right corner, the four texels around it.
  {
    hipTextureObject_t t = 0;
    hipTextureDesc d = tex_desc(hipAddressModeClamp, hipFilterModePoint, 1);
    CHECK(hipCreateTextureObject(&t, &res, &d, nullptr));
    for (int i = 0; i < n; ++i) at[i] = make_float2(static_cast<float>(i % W + 1) / W, static_cast<float>(i / W + 1) / H);
    (void)hipMemcpy(d_at, at.data(), n * sizeof(float2), hipMemcpyHostToDevice);
    gather2d<<<1, 64>>>(t, d_at, reinterpret_cast<float4*>(d_out), n, 1);
    std::vector<float4> r = host_copy(reinterpret_cast<float4*>(d_out), n);
    int wrong = 0;
    for (int i = 0; i < n; ++i) {
      const int x = i % W, y = i / W, x1 = clampi(x + 1, 0, W - 1), y1 = clampi(y + 1, 0, H - 1);
      const bool bad = r[i].x != f2d(x, y1) || r[i].y != f2d(x1, y1) || r[i].z != f2d(x1, y) || r[i].w != f2d(x, y);
      if (bad && !wrong)
        std::printf("  (%d, %d): %g %g %g %g, not %g %g %g %g\n", x, y, r[i].x, r[i].y, r[i].z, r[i].w, f2d(x, y1),
                    f2d(x1, y1), f2d(x1, y), f2d(x, y));
      wrong += bad;
    }
    // Component 0 is alpha, which a one-channel texture reads as 1.
    gather2d<<<1, 64>>>(t, d_at, reinterpret_cast<float4*>(d_out), n, 0);
    r = host_copy(reinterpret_cast<float4*>(d_out), n);
    for (int i = 0; i < n; ++i) wrong += r[i].x != 1 || r[i].y != 1 || r[i].z != 1 || r[i].w != 1;
    report("gathers", wrong, 2 * n);
    CHECK(hipDestroyTextureObject(t));
  }

  // ---- 8-bit texels, as normalized floats and as integers ----
  {
    const hipChannelFormatDesc u4 = hipCreateChannelDesc<uchar4>();
    hipArray_t a8 = nullptr;
    CHECK(hipMallocArray(&a8, &u4, W, H));
    std::vector<uchar4> px(n);
    for (int i = 0; i < n; ++i) px[i] = make_uchar4(i * 7, 255 - i, i * 3, 200);
    CHECK(hipMemcpy2DToArray(a8, 0, 0, px.data(), W * 4, W * 4, H, hipMemcpyHostToDevice));
    hipResourceDesc r8;
    std::memset(&r8, 0, sizeof r8);
    r8.resType = hipResourceTypeArray;
    r8.res.array.array = a8;
    hipTextureObject_t tf = 0, ti = 0;
    hipTextureDesc d = tex_desc(hipAddressModeClamp, hipFilterModePoint, 0, hipReadModeNormalizedFloat);
    CHECK(hipCreateTextureObject(&tf, &r8, &d, nullptr));
    d.readMode = hipReadModeElementType;
    CHECK(hipCreateTextureObject(&ti, &r8, &d, nullptr));
    for (int i = 0; i < n; ++i) at[i] = make_float2(i % W + 0.5f, i / W + 0.5f);
    (void)hipMemcpy(d_at, at.data(), n * sizeof(float2), hipMemcpyHostToDevice);
    sample2d_norm<<<1, 64>>>(tf, d_at, reinterpret_cast<float4*>(d_out), n);
    const std::vector<float4> rf = host_copy(reinterpret_cast<float4*>(d_out), n);
    int wrong = 0;
    for (int i = 0; i < n; ++i)
      wrong += std::fabs(rf[i].x - px[i].x / 255.0f) > 1e-6f || std::fabs(rf[i].y - px[i].y / 255.0f) > 1e-6f ||
               std::fabs(rf[i].z - px[i].z / 255.0f) > 1e-6f || std::fabs(rf[i].w - px[i].w / 255.0f) > 1e-6f;
    report("8-bit texels read as normalized floats", wrong, n);
    sample2d_uchar<<<1, 64>>>(ti, d_at, reinterpret_cast<uchar4*>(d_out), n);
    const std::vector<uchar4> ri = host_copy(reinterpret_cast<uchar4*>(d_out), n);
    wrong = 0;
    for (int i = 0; i < n; ++i)
      wrong += ri[i].x != px[i].x || ri[i].y != px[i].y || ri[i].z != px[i].z || ri[i].w != px[i].w;
    report("8-bit texels read as integers", wrong, n);
    // sRGB: the color channels come back linear, alpha as it is.
    hipTextureObject_t ts = 0;
    d.readMode = hipReadModeNormalizedFloat;
    d.sRGB = 1;
    CHECK(hipCreateTextureObject(&ts, &r8, &d, nullptr));
    sample2d_norm<<<1, 64>>>(ts, d_at, reinterpret_cast<float4*>(d_out), n);
    const std::vector<float4> rs = host_copy(reinterpret_cast<float4*>(d_out), n);
    const auto linear = [](unsigned char b) {
      const float c = b / 255.0f;
      return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    };
    wrong = 0;
    for (int i = 0; i < n; ++i)
      wrong += std::fabs(rs[i].x - linear(px[i].x)) > 1e-5f || std::fabs(rs[i].y - linear(px[i].y)) > 1e-5f ||
               std::fabs(rs[i].z - linear(px[i].z)) > 1e-5f || std::fabs(rs[i].w - px[i].w / 255.0f) > 1e-6f;
    report("8-bit sRGB texels read as linear floats", wrong, n);
    CHECK(hipDestroyTextureObject(ts));
    CHECK(hipDestroyTextureObject(tf));
    CHECK(hipDestroyTextureObject(ti));
    CHECK(hipFreeArray(a8));
  }

  // ---- Linear and pitched memory ----
  {
    std::vector<float> lin(64);
    for (int i = 0; i < 64; ++i) lin[i] = i * 1.5f - 7;
    float* d_lin = device_copy(lin);
    hipResourceDesc rl;
    std::memset(&rl, 0, sizeof rl);
    rl.resType = hipResourceTypeLinear;
    rl.res.linear.devPtr = d_lin;
    rl.res.linear.desc = fdesc;
    rl.res.linear.sizeInBytes = 64 * sizeof(float);
    hipTextureObject_t t = 0;
    hipTextureDesc d = tex_desc(hipAddressModeClamp, hipFilterModePoint, 0);
    CHECK(hipCreateTextureObject(&t, &rl, &d, nullptr));
    fetch1d<<<1, 64>>>(t, d_out, 64);
    std::vector<float> r = host_copy(d_out, 64);
    int wrong = 0;
    for (int i = 0; i < 64; ++i) wrong += r[i] != lin[i];
    report("tex1Dfetch from linear memory", wrong, 64);
    CHECK(hipDestroyTextureObject(t));

    // The driver API's form, two ints to an element, read as integers.
    std::vector<int2> pairs(32);
    for (int i = 0; i < 32; ++i) pairs[i] = make_int2(i * 3 - 40, -i);
    int2* d_pairs = device_copy(pairs);
    HIP_RESOURCE_DESC dr;
    std::memset(&dr, 0, sizeof dr);
    dr.resType = HIP_RESOURCE_TYPE_LINEAR;
    dr.res.linear.devPtr = d_pairs;
    dr.res.linear.format = HIP_AD_FORMAT_SIGNED_INT32;
    dr.res.linear.numChannels = 2;
    dr.res.linear.sizeInBytes = 32 * sizeof(int2);
    HIP_TEXTURE_DESC dt;
    std::memset(&dt, 0, sizeof dt);
    dt.flags = HIP_TRSF_READ_AS_INTEGER;
    hipTextureObject_t t2 = 0;
    CHECK(hipTexObjectCreate(&t2, &dr, &dt, nullptr));
    fetch1d_int2<<<1, 32>>>(t2, reinterpret_cast<int2*>(d_out), 32);
    const std::vector<int2> r2 = host_copy(reinterpret_cast<int2*>(d_out), 32);
    wrong = 0;
    for (int i = 0; i < 32; ++i) wrong += r2[i].x != pairs[i].x || r2[i].y != pairs[i].y;
    HIP_RESOURCE_DESC back;
    CHECK(hipTexObjectGetResourceDesc(&back, t2));
    wrong += back.res.linear.numChannels != 2 || back.res.linear.format != HIP_AD_FORMAT_SIGNED_INT32;
    report("hipTexObjectCreate over linear memory", wrong, 33);
    CHECK(hipTexObjectDestroy(t2));

    // Pitched memory: rows 256 bytes apart.
    float* d_pitched = nullptr;
    size_t pitch = 0;
    CHECK(hipMallocPitch(reinterpret_cast<void**>(&d_pitched), &pitch, W * sizeof(float), H));
    CHECK(hipMemcpy2D(d_pitched, pitch, texels.data(), W * sizeof(float), W * sizeof(float), H,
                      hipMemcpyHostToDevice));
    hipResourceDesc rp;
    std::memset(&rp, 0, sizeof rp);
    rp.resType = hipResourceTypePitch2D;
    rp.res.pitch2D.devPtr = d_pitched;
    rp.res.pitch2D.desc = fdesc;
    rp.res.pitch2D.width = W;
    rp.res.pitch2D.height = H;
    rp.res.pitch2D.pitchInBytes = pitch;
    CHECK(hipCreateTextureObject(&t, &rp, &d, nullptr));
    for (int i = 0; i < n; ++i) at[i] = make_float2(i % W + 0.5f, i / W + 0.5f);
    r = run(t, at);
    wrong = 0;
    for (int i = 0; i < n; ++i) wrong += r[i] != f2d(i % W, i / W);
    report("tex2D from pitched memory", wrong + (pitch % 256 != 0), n + 1);
    CHECK(hipDestroyTextureObject(t));
    CHECK(hipFree(d_lin));
    CHECK(hipFree(d_pairs));
    CHECK(hipFree(d_pitched));
  }

  // ---- Texture references ----
  {
    std::vector<float> lin(48);
    for (int i = 0; i < 48; ++i) lin[i] = i * 2.5f - 1;
    float* d_lin = device_copy(lin);
    size_t offset = 1;
    CHECK(hipBindTexture(&offset, g_linear_ref, d_lin, 48 * sizeof(float)));
    fetch_reference<<<1, 64>>>(d_out, 48);
    std::vector<float> r = host_copy(d_out, 48);
    int wrong = offset != 0;
    for (int i = 0; i < 48; ++i) wrong += r[i] != lin[i];
    CHECK(hipUnbindTexture(g_linear_ref));
    CHECK(hipBindTextureToArray(g_array_ref, arr, fdesc));
    sample_reference<<<1, 64>>>(d_out, W, H);
    r = host_copy(d_out, n);
    for (int i = 0; i < n; ++i) wrong += r[i] != f2d(i % W, i / W);
    hipArray_t bound = nullptr;
    const textureReference* ref = nullptr;
    CHECK(hipGetTextureReference(&ref, &g_array_ref));
    wrong += ref != &g_array_ref;
    CHECK(hipUnbindTexture(g_array_ref));
    wrong += hipTexRefGetArray(&bound, &g_array_ref) != hipErrorInvalidValue;   // bound to nothing now
    report("texture references bound to memory and to an array", wrong, 48 + n + 3);
    CHECK(hipFree(d_lin));
  }

  // ---- 3D and layered arrays, filled by hipMemcpy3D ----
  {
    constexpr int N = 4, D = 3;
    std::vector<float> box(N * N * D);
    for (size_t i = 0; i < box.size(); ++i) box[i] = static_cast<float>(i) * 0.5f + 1;
    const auto fill = [&](hipArray_t a, int w, int h, int d) {
      hipMemcpy3DParms c;
      std::memset(&c, 0, sizeof c);
      c.srcPtr = make_hipPitchedPtr(box.data(), w * sizeof(float), w, h ? h : 1);
      c.dstArray = a;
      c.extent = make_hipExtent(w, h ? h : 1, d);
      c.kind = hipMemcpyHostToDevice;
      return hipMemcpy3D(&c);
    };
    hipArray_t a3 = nullptr, a2l = nullptr, a1l = nullptr;
    CHECK(hipMalloc3DArray(&a3, &fdesc, make_hipExtent(N, N, D), 0));
    CHECK(hipMalloc3DArray(&a2l, &fdesc, make_hipExtent(N, N, D), hipArrayLayered));
    CHECK(hipMalloc3DArray(&a1l, &fdesc, make_hipExtent(N * N, 0, D), hipArrayLayered));
    CHECK(fill(a3, N, N, D));
    CHECK(fill(a2l, N, N, D));
    CHECK(fill(a1l, N * N, 0, D));
    const auto texture = [&](hipArray_t a) {
      hipResourceDesc r;
      std::memset(&r, 0, sizeof r);
      r.resType = hipResourceTypeArray;
      r.res.array.array = a;
      hipTextureDesc d = tex_desc(hipAddressModeClamp, hipFilterModePoint, 0);
      hipTextureObject_t t = 0;
      (void)hipCreateTextureObject(&t, &r, &d, nullptr);
      return t;
    };
    const int count = N * N * D;
    int wrong = 0;
    hipTextureObject_t t = texture(a3);
    sample3d<<<1, 64>>>(t, d_out, N, D);
    std::vector<float> r = host_copy(d_out, count);
    for (int i = 0; i < count; ++i) wrong += r[i] != box[i];
    report("tex3D from a 3D array", wrong, count);
    CHECK(hipDestroyTextureObject(t));
    wrong = 0;
    t = texture(a2l);
    sample2d_layered<<<1, 64>>>(t, d_out, N, D);
    r = host_copy(d_out, count);
    for (int i = 0; i < count; ++i) wrong += r[i] != box[i];
    report("tex2DLayered from a 2D layered array", wrong, count);
    CHECK(hipDestroyTextureObject(t));
    wrong = 0;
    t = texture(a1l);
    sample1d_layered<<<1, 64>>>(t, d_out, N * N, D);
    r = host_copy(d_out, count);
    for (int i = 0; i < count; ++i) wrong += r[i] != box[i];
    report("tex1DLayered from a 1D layered array", wrong, count);
    CHECK(hipDestroyTextureObject(t));
    // And back out of the 3D array.
    std::vector<float> out(box.size(), -1);
    hipMemcpy3DParms c;
    std::memset(&c, 0, sizeof c);
    c.srcArray = a3;
    c.dstPtr = make_hipPitchedPtr(out.data(), N * sizeof(float), N, N);
    c.extent = make_hipExtent(N, N, D);
    c.kind = hipMemcpyDeviceToHost;
    CHECK(hipMemcpy3D(&c));
    wrong = 0;
    for (size_t i = 0; i < box.size(); ++i) wrong += out[i] != box[i];
    report("a 3D array copied out", wrong, count);
    // A batch: the 3D array from its second column into a pointer whose rows
    // are elements long, the array into the layered one, and that one out.
    std::vector<float> part(static_cast<size_t>((N - 1) * N * D), -1), whole(box.size(), -1);
    hipMemcpy3DBatchOp ops[3];
    std::memset(ops, 0, sizeof ops);
    ops[0].src.type = hipMemcpyOperandTypeArray;
    ops[0].src.op.array.array = a3;
    ops[0].src.op.array.offset = {1, 0, 0};
    ops[0].dst.type = hipMemcpyOperandTypePointer;
    ops[0].dst.op.ptr.ptr = part.data();
    ops[0].dst.op.ptr.rowLength = N - 1;
    ops[0].extent = make_hipExtent(N - 1, N, D);
    ops[1].src.type = ops[1].dst.type = hipMemcpyOperandTypeArray;
    ops[1].src.op.array.array = a3;
    ops[1].dst.op.array.array = a2l;
    ops[1].extent = make_hipExtent(N, N, D);
    ops[2].src.type = hipMemcpyOperandTypeArray;
    ops[2].src.op.array.array = a2l;
    ops[2].dst.type = hipMemcpyOperandTypePointer;
    ops[2].dst.op.ptr.ptr = whole.data();
    ops[2].extent = make_hipExtent(N, N, D);
    for (auto& op : ops) op.srcAccessOrder = hipMemcpySrcAccessOrderStream;
    size_t failed = 0;
    CHECK(hipMemcpy3DBatchAsync(3, ops, &failed, 0, nullptr));
    CHECK(hipStreamSynchronize(nullptr));
    wrong = failed != SIZE_MAX;
    for (int z = 0; z < D; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N - 1; ++x) wrong += part[(z * N + y) * (N - 1) + x] != box[(z * N + y) * N + x + 1];
    for (size_t i = 0; i < box.size(); ++i) wrong += whole[i] != box[i];
    report("a batch of 3D copies to, from and between arrays", wrong, 1 + (N - 1) * N * D + count);
    CHECK(hipFreeArray(a3));
    CHECK(hipFreeArray(a2l));
    CHECK(hipFreeArray(a1l));
  }

  // ---- A surface, written and read ----
  {
    hipArray_t as = nullptr;
    CHECK(hipMallocArray(&as, &fdesc, W, H, hipArraySurfaceLoadStore));
    hipResourceDesc rs;
    std::memset(&rs, 0, sizeof rs);
    rs.resType = hipResourceTypeArray;
    rs.res.array.array = as;
    hipSurfaceObject_t s = 0;
    CHECK(hipCreateSurfaceObject(&s, &rs));
    surface_write<<<H, W>>>(s);
    surface_read<<<H, W>>>(s, d_out);
    const std::vector<float> r = host_copy(d_out, n);
    std::vector<float> back(n, -1);
    CHECK(hipMemcpy2DFromArray(back.data(), W * sizeof(float), as, 0, 0, W * sizeof(float), H,
                               hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int i = 0; i < n; ++i) {
      const float want = static_cast<float>((i % W) * 100 + i / W);
      wrong += r[i] != want || back[i] != want;
    }
    surface_cube_face<<<H, W>>>(s, d_out);
    const std::vector<float> rc = host_copy(d_out, n);
    for (int i = 0; i < n; ++i) wrong += rc[i] != static_cast<float>(i % W - i / W);
    report("a surface written and read, and through a cube face", wrong, 2 * n);
    CHECK(hipDestroySurfaceObject(s));
    CHECK(hipFreeArray(as));
  }

  // ---- The API's answers ----
  {
    int wrong = 0;
    hipChannelFormatDesc got;
    hipExtent e;
    unsigned int flags = 99;
    CHECK(hipGetChannelDesc(&got, arr));
    CHECK(hipArrayGetInfo(nullptr, &e, &flags, arr));
    wrong += got.x != 32 || got.y != 0 || got.f != hipChannelFormatKindFloat;
    wrong += e.width != W || e.height != H || e.depth != 0 || flags != 0;
    hipTextureObject_t t = 0;
    hipTextureDesc d = tex_desc(hipAddressModeClamp, hipFilterModePoint, 1);
    wrong += hipCreateTextureObject(nullptr, &res, &d, nullptr) != hipErrorInvalidChannelDescriptor;
    CHECK(hipCreateTextureObject(&t, &res, &d, nullptr));
    hipResourceDesc rback;
    CHECK(hipGetTextureObjectResourceDesc(&rback, t));
    wrong += rback.resType != hipResourceTypeArray || rback.res.array.array != arr;
    CHECK(hipDestroyTextureObject(t));
    hipArray_t bad = nullptr;
    const hipChannelFormatDesc three = hipCreateChannelDesc(32, 32, 32, 0, hipChannelFormatKindFloat);
    wrong += hipMallocArray(&bad, &three, W, H) != hipErrorInvalidValue;
    hipArray_t layered = nullptr;
    CHECK(hipMalloc3DArray(&layered, &fdesc, make_hipExtent(W, H, 2), hipArrayLayered));
    hipResourceDesc rl;
    std::memset(&rl, 0, sizeof rl);
    rl.resType = hipResourceTypeArray;
    rl.res.array.array = layered;
    hipSurfaceObject_t s = 0;
    wrong += hipCreateSurfaceObject(&s, &rl) != hipErrorInvalidValue;   // not made for loads and stores
    CHECK(hipFreeArray(layered));
    wrong += hipFreeArray(layered) != hipErrorContextIsDestroyed;
    // A 1D array through the driver API's byte copies.
    hipArray_t a1 = nullptr;
    HIP_ARRAY_DESCRIPTOR ad;
    std::memset(&ad, 0, sizeof ad);
    ad.Width = 16;
    ad.Format = HIP_AD_FORMAT_UNSIGNED_INT16;
    ad.NumChannels = 2;
    CHECK(hipArrayCreate(&a1, &ad));
    std::vector<uint16_t> words(32), again(32, 0);
    for (int i = 0; i < 32; ++i) words[i] = static_cast<uint16_t>(i * 1000 + 7);
    CHECK(hipMemcpyHtoA(a1, 0, words.data(), 64));
    CHECK(hipMemcpyAtoH(again.data(), a1, 0, 64));
    for (int i = 0; i < 32; ++i) wrong += words[i] != again[i];
    HIP_ARRAY_DESCRIPTOR adback;
    CHECK(hipArrayGetDescriptor(&adback, a1));
    wrong += adback.Width != 16 || adback.NumChannels != 2 || adback.Format != HIP_AD_FORMAT_UNSIGNED_INT16;
    CHECK(hipArrayDestroy(a1));
    report("what the API answers", wrong, 44);
  }
  CHECK(hipFreeArray(arr));
  CHECK(hipFree(d_at));
  CHECK(hipFree(d_out));
  return 0;
}
