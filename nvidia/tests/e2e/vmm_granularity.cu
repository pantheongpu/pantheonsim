// The virtual memory API's granularity and what each call does with a size, an offset or an alignment that is
// not a multiple of it (nvidia/tests/data/vmm_granularity.rtx3060.expected is what an RTX 3060 printed).
// cuMemGetAllocationGranularity answers 2 MiB, the minimum and the recommended one alike, whatever the
// property says (the location type and device are not even checked); the memory the calls hand out is in
// units of it. The program prints the answers, never an address.
#include <cuda.h>
#include <cstdio>
#include <cstring>

static const char* name(CUresult r) {
  const char* n = nullptr;
  cuGetErrorName(r, &n);
  return n ? n : "?";
}
#define SHOW(label, expr) do { const CUresult r_ = (expr); std::printf("%-58s %s\n", label, name(r_)); } while (0)

int main() {
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { std::printf("FAIL: no device\n"); return 1; }
#if CUDA_VERSION >= 13000
  if (cuCtxCreate(&ctx, nullptr, 0, dev) != CUDA_SUCCESS) { std::printf("FAIL: no context\n"); return 1; }
#else
  if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) { std::printf("FAIL: no context\n"); return 1; }
#endif
  CUmemAllocationProp prop;
  std::memset(&prop, 0, sizeof prop);
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;

  // the granularity, for every kind of property
  size_t g = 0;
  auto gran = [&](const char* label, const CUmemAllocationProp* p, int option) {
    size_t v = 1234;
    const CUresult r = cuMemGetAllocationGranularity(&v, p, static_cast<CUmemAllocationGranularity_flags>(option));
    std::printf("%-58s %s %zu\n", label, name(r), r == CUDA_SUCCESS ? v : size_t{0});
  };
  gran("granularity minimum", &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  gran("granularity recommended", &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  CUmemAllocationProp q = prop;
  q.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  gran("granularity minimum, exportable as a descriptor", &q, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  gran("granularity recommended, exportable as a descriptor", &q, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  q = prop;
  q.location.type = static_cast<CUmemLocationType>(2);   // CU_MEM_LOCATION_TYPE_HOST (CUDA 12.2 and later name it)
  gran("granularity minimum, host location", &q, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  q = prop;
  q.location.type = static_cast<CUmemLocationType>(99);
  gran("granularity minimum, location type 99", &q, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  q = prop;
  q.location.id = 7;
  gran("granularity minimum, device 7", &q, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  q = prop;
  q.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
  gran("granularity recommended, compressible", &q, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  gran("granularity, option 2", &prop, 2);
  gran("granularity, option -1", &prop, -1);
  gran("granularity, no property", nullptr, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  {
    const CUresult r = cuMemGetAllocationGranularity(nullptr, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    std::printf("%-58s %s\n", "granularity, no output", name(r));
  }
  CUresult rc = cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (rc != CUDA_SUCCESS || g == 0) { std::printf("FAIL: no granularity\n"); return 1; }

  // cuMemCreate: sizes
  const size_t sizes[] = {1, 4096, 65536, g / 2, g - 1, g, g + 1, 2 * g, 3 * g / 2, 3 * g};
  for (const size_t s : sizes) {
    CUmemGenericAllocationHandle h = 0;
    char label[80];
    std::snprintf(label, sizeof label, "cuMemCreate %zu", s);
    const CUresult r = cuMemCreate(&h, s, &prop, 0);
    SHOW(label, r);
    if (r == CUDA_SUCCESS) cuMemRelease(h);
  }
  { CUmemGenericAllocationHandle h = 0; SHOW("cuMemCreate 0", cuMemCreate(&h, 0, &prop, 0)); }

  // cuMemAddressReserve: sizes and alignments; what the alignment of the address is (the largest power of two
  // up to 1 GiB that divides it)
  auto reserve = [&](const char* label, size_t size, size_t align) {
    CUdeviceptr p = 0;
    const CUresult r = cuMemAddressReserve(&p, size, align, 0, 0);
    int a = 0;
    if (r == CUDA_SUCCESS) {
      while (a < 30 && (p & ((CUdeviceptr{1} << (a + 1)) - 1)) == 0) ++a;
      cuMemAddressFree(p, size);
    }
    std::printf("%-58s %s%s%s\n", label, name(r), r == CUDA_SUCCESS ? " aligned to at least 2^" : "",
                r == CUDA_SUCCESS ? (a >= 21 ? "21" : (a >= 16 ? "16" : "0")) : "");
  };
  reserve("reserve 1", 1, 0);
  reserve("reserve 4096", 4096, 0);
  reserve("reserve 65536", 65536, 0);
  reserve("reserve g", g, 0);
  reserve("reserve g + 1", g + 1, 0);
  reserve("reserve 2g", 2 * g, 0);
  reserve("reserve 3g/2", 3 * g / 2, 0);
  reserve("reserve g, alignment 4096", g, 4096);
  reserve("reserve g, alignment 64 KiB", g, 65536);
  reserve("reserve g, alignment g", g, g);
  reserve("reserve g, alignment 2g", g, 2 * g);
  reserve("reserve g, alignment 3g", g, 3 * g);
  reserve("reserve g, alignment 1 GiB", g, size_t{1} << 30);
  reserve("reserve g, alignment 5", g, 5);
  { CUdeviceptr p = 0; SHOW("reserve 0", cuMemAddressReserve(&p, 0, 0, 0, 0)); }
  { CUdeviceptr p = 0; SHOW("reserve with flags", cuMemAddressReserve(&p, g, 0, 0, 1)); }
  { SHOW("reserve with no output", cuMemAddressReserve(nullptr, g, 0, 0, 0)); }

  // map, access, unmap and free on a 4g reservation and a 2g handle
  CUdeviceptr base = 0;
  CUmemGenericAllocationHandle h = 0;
  if (cuMemAddressReserve(&base, 4 * g, 0, 0, 0) != CUDA_SUCCESS || cuMemCreate(&h, 2 * g, &prop, 0) != CUDA_SUCCESS) {
    std::printf("FAIL: setup\n");
    return 1;
  }
  // maps that fail change nothing
  SHOW("map g/2", cuMemMap(base, g / 2, 0, h, 0));
  SHOW("map g + 1", cuMemMap(base, g + 1, 0, h, 0));
  SHOW("map 0", cuMemMap(base, 0, 0, h, 0));
  SHOW("map at base + 4096", cuMemMap(base + 4096, g, 0, h, 0));
  SHOW("map at base + 64 KiB", cuMemMap(base + 65536, g, 0, h, 0));
  SHOW("map offset 4096", cuMemMap(base, g, 4096, h, 0));
  SHOW("map offset 64 KiB", cuMemMap(base, g, 65536, h, 0));
  SHOW("map offset 2g (the end of the handle)", cuMemMap(base, g, 2 * g, h, 0));
  SHOW("map 3g of a 2g handle", cuMemMap(base, 3 * g, 0, h, 0));
  SHOW("map g at offset g + g/2", cuMemMap(base, g, g + g / 2, h, 0));
  SHOW("map with flags", cuMemMap(base, g, 0, h, 1));
  SHOW("map with no handle", cuMemMap(base, g, 0, 0, 0));
  SHOW("map 2g", cuMemMap(base, 2 * g, 0, h, 0));
  SHOW("map again over it", cuMemMap(base, g, 0, h, 0));
  SHOW("map g at the second g", cuMemMap(base + g, g, 0, h, 0));
  CUmemAccessDesc rw;
  std::memset(&rw, 0, sizeof rw);
  rw.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  rw.location.id = 0;
  rw.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  SHOW("access g/2", cuMemSetAccess(base, g / 2, &rw, 1));
  SHOW("access g + 1", cuMemSetAccess(base, g + 1, &rw, 1));
  SHOW("access 0", cuMemSetAccess(base, 0, &rw, 1));
  SHOW("access at base + 4096", cuMemSetAccess(base + 4096, g, &rw, 1));
  SHOW("access at base + 64 KiB", cuMemSetAccess(base + 65536, g, &rw, 1));
  SHOW("access past the mapping", cuMemSetAccess(base + 2 * g, g, &rw, 1));
  SHOW("access with no descriptor", cuMemSetAccess(base, g, nullptr, 1));
  SHOW("access with a count of 0", cuMemSetAccess(base, g, &rw, 0));
  SHOW("access the first g of the 2g mapping", cuMemSetAccess(base, g, &rw, 1));
  SHOW("access 2g", cuMemSetAccess(base, 2 * g, &rw, 1));
  // unmaps that fail
  SHOW("unmap g/2", cuMemUnmap(base, g / 2));
  SHOW("unmap g + 1", cuMemUnmap(base, g + 1));
  SHOW("unmap 0", cuMemUnmap(base, 0));
  SHOW("unmap at base + 4096", cuMemUnmap(base + 4096, g));
  SHOW("unmap at base + 64 KiB", cuMemUnmap(base + 65536, g));
  SHOW("unmap the second g of the 2g mapping", cuMemUnmap(base + g, g));
  SHOW("unmap the first g of the 2g mapping", cuMemUnmap(base, g));
  SHOW("free the reservation while mapped", cuMemAddressFree(base, 4 * g));
  SHOW("unmap 2g", cuMemUnmap(base, 2 * g));
  SHOW("unmap 2g again", cuMemUnmap(base, 2 * g));
  // frees that fail
  SHOW("free g/2", cuMemAddressFree(base, g / 2));
  SHOW("free g", cuMemAddressFree(base, g));
  SHOW("free 4g + 1", cuMemAddressFree(base, 4 * g + 1));
  SHOW("free at base + 4096", cuMemAddressFree(base + 4096, 4 * g));
  SHOW("free at base + g", cuMemAddressFree(base + g, 3 * g));
  SHOW("free 5g", cuMemAddressFree(base, 5 * g));
  SHOW("free 0", cuMemAddressFree(base, 0));
  SHOW("free 4g", cuMemAddressFree(base, 4 * g));
  SHOW("free 4g again", cuMemAddressFree(base, 4 * g));
  SHOW("release the handle", cuMemRelease(h));
  SHOW("release the handle again", cuMemRelease(h));
  std::printf("PASS\n");
  return 0;
}
