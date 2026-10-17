// Where cudaMalloc puts things: every pointer it returns is as aligned as a texture asks
// (cudaDeviceProp::textureAlignment, 512 bytes), whatever was allocated before it, so a linear texture can be
// made over any allocation with no offset. NVIDIA's runtime documents this ("if the pointer came from
// cudaMalloc, the offset is 0"), and CUDA Samples rely on it: convolutionFFT2D runs its tests one after the
// other, each allocating a 168-byte kernel among larger buffers and making a texture over a buffer it
// allocated first.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

// A texture over [p, p + bytes) as floats, as convolutionFFT2D makes one.
static cudaError_t make_texture(void* p, size_t bytes) {
  cudaResourceDesc res = {};
  res.resType = cudaResourceTypeLinear;
  res.res.linear.devPtr = p;
  res.res.linear.sizeInBytes = bytes;
  res.res.linear.desc = cudaCreateChannelDesc<float>();
  cudaTextureDesc tex = {};
  tex.filterMode = cudaFilterModeLinear;
  tex.addressMode[0] = cudaAddressModeWrap;
  tex.readMode = cudaReadModeElementType;
  cudaTextureObject_t t = 0;
  const cudaError_t e = cudaCreateTextureObject(&t, &res, &tex, nullptr);
  if (e == cudaSuccess) cudaDestroyTextureObject(t);
  return e;
}

int main() {
  cudaFree(0);
  int alignment = 0;
  cudaDeviceGetAttribute(&alignment, cudaDevAttrTextureAlignment, 0);
  check(alignment == 512, "the texture alignment is 512 bytes");

  // Sizes that are not multiples of 512 leave the next pointer short of it unless each allocation is
  // rounded up: 1, 4, 168 (a 7 x 6 float kernel), 256, 1000 and a few round and odd ones.
  const size_t sizes[] = {1, 4, 168, 256, 1000, 4096, 100000, 16000000, 168, 33};
  std::vector<void*> ptrs;
  bool aligned = true, textured = true;
  for (size_t bytes : sizes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { aligned = false; break; }
    ptrs.push_back(p);
    if (reinterpret_cast<uintptr_t>(p) % 512) aligned = false;
    // A texture holds whole texels: the floats that fit.
    if (bytes >= 4 && make_texture(p, bytes / 4 * 4) != cudaSuccess) textured = false;
  }
  check(aligned, "every cudaMalloc pointer is a multiple of 512, whatever came before it");
  check(textured, "a linear texture can be made over each one");
  for (void* p : ptrs) cudaFree(p);

  // The same after everything was freed: the next allocations come after a run that left the allocator at an
  // address that is not a multiple of 512 plus the size of its last block.
  void* a = nullptr;
  void* b = nullptr;
  void* c = nullptr;
  check(cudaMalloc(&a, 168) == cudaSuccess && cudaMalloc(&b, 16000000) == cudaSuccess, "168 and then 16000000 bytes");
  cudaFree(a);
  cudaFree(b);
  check(cudaMalloc(&c, 16000000) == cudaSuccess && make_texture(c, 16000000) == cudaSuccess,
        "a 16000000-byte buffer allocated afterwards takes a texture too");
  cudaFree(c);

  // The pitched allocation comes from the same allocator.
  void* pitched = nullptr;
  size_t pitch = 0;
  check(cudaMallocPitch(&pitched, &pitch, 100, 7) == cudaSuccess && reinterpret_cast<uintptr_t>(pitched) % 512 == 0,
        "cudaMallocPitch's pointer is a multiple of 512");
  cudaFree(pitched);

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
