// malloc() and free() in a kernel draw on one heap per device, of
// cudaLimitMallocHeapSize bytes, which lasts until the device is reset:
//  - blocks leaked before cudaDeviceReset do not count after it, so a kernel
//    can then have nearly the whole heap again;
//  - free() gives the bytes back, within a kernel and across kernels;
//  - a block one kernel allocates, a later kernel may read and free;
//  - the host's cudaFree refuses a block a kernel allocated, which stays live,
//    and so do the driver's cuMemFree and cuMemGetAddressRange;
//  - and the same across the simulator's two engines: device_heap_peer.cu is
//    built to PTX only, so its kernels run on the PTX interpreter while these
//    run as SASS, and each frees what the other allocated.
// Checked against an RTX 3060, where a single malloc can have 7 of the default
// 8 MiB (the allocator keeps some of it) but not 7.5.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
#define CHECK(c) do { if (!(c)) { printf("FAIL %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

void launch_peer_keep(void** slot, size_t bytes, unsigned char mark);
void launch_peer_check_free(void** slot, size_t bytes, int* seen);
void launch_peer_try(size_t bytes, int* ok);

// Thread i allocates `bytes`, touches both ends, and never frees it.
__global__ void leak(size_t bytes, int* ok) {
  unsigned char* p = static_cast<unsigned char*>(malloc(bytes));
  ok[threadIdx.x] = p != nullptr;
  if (p) {
    p[0] = 1;
    p[bytes - 1] = 1;
  }
}

// Whether `bytes` can be had now; given back at once.
__global__ void try_alloc(size_t bytes, int* ok) {
  unsigned char* p = static_cast<unsigned char*>(malloc(bytes));
  *ok = p != nullptr;
  if (p) p[bytes - 1] = 1;
  free(p);
}

// Most of the heap, freed, and then most of it again in the same thread.
__global__ void alloc_free_alloc(size_t bytes, int* ok) {
  void* p = malloc(bytes);
  ok[0] = p != nullptr;
  free(p);
  void* q = malloc(bytes);
  ok[1] = q != nullptr;
  free(q);
}

// A block kept past the kernel, marked at both ends, for a later one.
__global__ void keep(void** slot, size_t bytes, unsigned char mark) {
  unsigned char* p = static_cast<unsigned char*>(malloc(bytes));
  if (p) {
    p[0] = mark;
    p[bytes - 1] = mark;
  }
  *slot = p;
}

// Reads the marks of a block an earlier kernel kept, and frees it.
__global__ void check_free(void** slot, size_t bytes, int* seen) {
  unsigned char* p = static_cast<unsigned char*>(*slot);
  *seen = p ? p[0] | p[bytes - 1] << 8 : -1;
  free(p);
  *slot = nullptr;
}

struct Buffers {
  int* ok = nullptr;      // device results
  void** slot = nullptr;  // a block pointer handed between kernels
};

static cudaError_t make(Buffers& b) {
  cudaError_t e = cudaMalloc(&b.ok, 16 * sizeof(int));
  if (e == cudaSuccess) e = cudaMemset(b.ok, 0, 16 * sizeof(int));
  if (e == cudaSuccess) e = cudaMalloc(&b.slot, sizeof(void*));
  if (e == cudaSuccess) e = cudaMemset(b.slot, 0, sizeof(void*));
  return e;
}

// The first `n` results, after the launches before it have finished.
static bool results(const Buffers& b, int* out, int n) {
  if (cudaDeviceSynchronize() != cudaSuccess) return false;
  return cudaMemcpy(out, b.ok, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
}

int main() {
  size_t heap = 0;
  CK(cudaDeviceGetLimit(&heap, cudaLimitMallocHeapSize));
  CHECK(heap == size_t{8} << 20);
  const size_t most = heap / 8 * 7;   // what one malloc can still have
  int r[16];

  // ---- leaked blocks do not outlive a reset ---------------------------------
  {
    Buffers b;
    CK(make(b));
    leak<<<1, 4>>>(heap / 8, b.ok);   // half the heap, never freed
    CHECK(results(b, r, 4));
    CHECK(r[0] && r[1] && r[2] && r[3]);
    try_alloc<<<1, 1>>>(most, b.ok);
    CHECK(results(b, r, 1));
    CHECK(r[0] == 0);   // the leaked half still counts...
  }
  CK(cudaDeviceReset());
  {
    size_t after = 0;
    CK(cudaDeviceGetLimit(&after, cudaLimitMallocHeapSize));
    CHECK(after == heap);
    Buffers b;
    CK(make(b));
    try_alloc<<<1, 1>>>(most, b.ok);
    CHECK(results(b, r, 1));
    if (r[0] != 1) {   // ...and after the reset it does not
      printf("FAIL a malloc of %zu of the %zu-byte heap after cudaDeviceReset got null: blocks "
             "leaked before the reset still count\n", most, heap);
      return 1;
    }
  }

  // ---- free() gives the bytes back ------------------------------------------
  Buffers b;
  CK(make(b));
  alloc_free_alloc<<<1, 1>>>(most, b.ok);
  CHECK(results(b, r, 2));
  CHECK(r[0] == 1 && r[1] == 1);
  try_alloc<<<1, 1>>>(most, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 1);

  // ---- a block outlives its kernel, and a later one frees it ------------------
  keep<<<1, 1>>>(b.slot, most, 0x5a);
  try_alloc<<<1, 1>>>(most, b.ok);   // the kept block is still held
  CHECK(results(b, r, 1));
  CHECK(r[0] == 0);
  check_free<<<1, 1>>>(b.slot, most, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 0x5a5a);
  try_alloc<<<1, 1>>>(most, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 1);

  // ---- one heap for both engines ----------------------------------------------
  // Kept here, freed by the PTX module.
  keep<<<1, 1>>>(b.slot, most, 0x3c);
  launch_peer_try(most, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 0);
  launch_peer_check_free(b.slot, most, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 0x3c3c);
  try_alloc<<<1, 1>>>(most, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 1);
  // Kept by the PTX module, freed here.
  launch_peer_keep(b.slot, most, 0x71);
  try_alloc<<<1, 1>>>(most, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 0);
  check_free<<<1, 1>>>(b.slot, most, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 0x7171);
  launch_peer_try(most, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 1);

  // ---- the host may not free a device-heap block --------------------------------
  // An RTX 3060's cudaFree of a block a kernel's malloc() handed out returns
  // InvalidValue, sets the last error, and leaves the block live: a later
  // kernel still reads its marks and frees it.
  keep<<<1, 1>>>(b.slot, 256, 0x5a);
  CK(cudaDeviceSynchronize());
  void* held = nullptr;
  CK(cudaMemcpy(&held, b.slot, sizeof held, cudaMemcpyDeviceToHost));
  CHECK(held != nullptr);
  CHECK(cudaFree(held) == cudaErrorInvalidValue);
  CHECK(cudaGetLastError() == cudaErrorInvalidValue);
  // The driver, on the card: cuMemFree refuses it with CUDA_ERROR_INVALID_VALUE,
  // and cuMemGetAddressRange answers CUDA_ERROR_NOT_FOUND, the block being the
  // heap's rather than an allocation of the host's. Neither frees it.
  const CUdeviceptr dheld = reinterpret_cast<CUdeviceptr>(held);
  CHECK(cuMemFree(dheld) == CUDA_ERROR_INVALID_VALUE);
  CUdeviceptr base = 0;
  size_t size = 0;
  CHECK(cuMemGetAddressRange(&base, &size, dheld) == CUDA_ERROR_NOT_FOUND);
  CHECK(cudaGetLastError() == cudaSuccess);   // the driver's refusals leave the runtime's error alone
  check_free<<<1, 1>>>(b.slot, 256, b.ok);
  CHECK(results(b, r, 1));
  CHECK(r[0] == 0x5a5a);

  CK(cudaGetLastError());
  printf("PASS\n");
  return 0;
}
