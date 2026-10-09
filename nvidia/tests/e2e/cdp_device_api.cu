// The device runtime as CUDA Samples' dynamic-parallelism programs use it
// (cdpSimplePrint, cdpSimpleQuicksort, cdpQuadtree), built as they are built:
// -rdc=true, linked with cudadevrt. On the SASS engine the device runtime's
// entry points are builtins (src/sass/module.cpp), and the linked cubin
// carries relocations a plain build does not: a kernel's function descriptor
// in a UMOV pair, and in the device runtime library's own tables.
//
// 1. printf from a grid launched by a grid launched by the host, recursing
//    as cdpSimplePrint does (each thread of each level launches the next).
// 2. A recursive sort, each half in a device-created stream (cdpSimpleQuicksort).
// 3. A child with dynamic shared memory (cdpQuadtree).
// 4. The device-side last error: a launch with too many threads per block
//    fails with cudaErrorInvalidConfiguration, which cudaPeekAtLastError
//    reports and leaves and cudaGetLastError reports and clears (0 9 9 0 on
//    an RTX 3060, CUDA 13.0).
// 5. cudaGetDevice and cudaGetDeviceCount in a child grid.
// 6. Tail launches, fire-and-forget launches and named streams: a tail
//    launch runs after the grid and everything it launched, in order.
// 7. cudaMemcpyAsync, cudaMemcpy2DAsync, cudaMemcpy3DAsync and the memset
//    family from a kernel, in their stream's order, and their errors.
// 8. cudaMalloc and cudaFree from a kernel (the device heap) and their errors.
// 9. cudaFuncGetAttributes, cudaDeviceGetAttribute, cudaDeviceGetLimit,
//    the cache configuration, cudaOccupancyMaxActiveBlocksPerMultiprocessor
//    and cudaGetErrorString/Name in a kernel, each agreeing with the host's.
// 10. Device streams and events: what their calls accept and refuse.
// 11. Launch configurations a child may not have, and launching into streams
//     that are none.
// 12. The pending-launch limit (cudaLimitDevRuntimePendingLaunchCount): a
//     chain of grids stops at the limit with cudaErrorLaunchPendingCountExceeded,
//     and more launches than the limit run when earlier ones finish.
// 13. The older cudaGetParameterBuffer / cudaLaunchDevice pair.
// 14. The last error is each thread's own.
//
// Every result is checked exactly; the output does not depend on the order
// grids run in. Each call's results were measured on an RTX 3060 (CUDA 13.0
// and 12.0), and this program passes there and on both engines. Prints PASS
// on the last line.
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      std::printf("FAIL: %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);   \
      return false;                                                                    \
    }                                                                                  \
  } while (0)

// 1. Each level's threads launch the next level; level `depth` counts its
// blocks, and only the deepest level prints, once.
__global__ void levels(int* count, int depth, int max_depth) {
  if (threadIdx.x == 0) atomicAdd(&count[depth], 1);
  if (depth + 1 < max_depth) {
    levels<<<2, 2>>>(count, depth + 1, max_depth);
  } else if (threadIdx.x == 0 && blockIdx.x == 0 && atomicAdd(&count[max_depth], 1) == 0) {
    printf("deepest level %d reached\n", depth);
  }
}

// 2. Quicksort with one thread per launch, as cdpSimpleQuicksort.
__global__ void quicksort(unsigned* data, int left, int right, int depth) {
  if (depth >= 16 || right - left <= 8) {
    for (int i = left; i <= right; ++i)
      for (int j = i + 1; j <= right; ++j)
        if (data[j] < data[i]) {
          unsigned t = data[i];
          data[i] = data[j];
          data[j] = t;
        }
    return;
  }
  unsigned* lptr = data + left;
  unsigned* rptr = data + right;
  const unsigned pivot = data[(left + right) / 2];
  while (lptr <= rptr) {
    unsigned lval = *lptr, rval = *rptr;
    while (lval < pivot) lval = *++lptr;
    while (rval > pivot) rval = *--rptr;
    if (lptr <= rptr) {
      *lptr++ = rval;
      *rptr-- = lval;
    }
  }
  const int nright = static_cast<int>(rptr - data), nleft = static_cast<int>(lptr - data);
  if (left < nright) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
    cudaStreamDestroy(s);
  }
  if (nleft < right) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    quicksort<<<1, 1, 0, s>>>(data, nleft, right, depth + 1);
    cudaStreamDestroy(s);
  }
}

// 3. Dynamic shared memory in a child: each block sums its slice through it.
__global__ void block_sums(const int* in, int* out) {
  extern __shared__ int part[];
  part[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    int s = 0;
    for (unsigned i = 0; i < blockDim.x; ++i) s += part[i];
    out[blockIdx.x] = s;
  }
}
__global__ void launch_sums(const int* in, int* out) { block_sums<<<4, 64, 64 * sizeof(int)>>>(in, out); }

// 4 and 5.
__global__ void noop() {}
__global__ void ask_device(int* out) {
  int dev = -1, count = -1;
  out[4] = cudaGetDevice(&dev);
  out[5] = dev;
  out[6] = cudaGetDeviceCount(&count);
  out[7] = count;
}
__global__ void errors(int* out) {
  out[0] = cudaGetLastError();        // nothing yet
  noop<<<1, 4096>>>();                // more threads than a block may have
  out[1] = cudaPeekAtLastError();     // reported, kept
  out[2] = cudaGetLastError();        // reported, cleared
  out[3] = cudaGetLastError();
  ask_device<<<1, 1>>>(out);
}


// ---- 6. tail launches, fire-and-forget launches, streams ----------------------------
//
// Each test kernel is launched alone; the flags X, Y, Z (flag[0..2]) record what had happened when
// read_flags ran, as the digits of X*100 + Y*10 + Z. nap_set takes a while before it sets its flag,
// so a launch that ran too early would read 0.
__device__ void nap() {
  for (int i = 0; i < 4; ++i) __nanosleep(500000);
}
__global__ void nap_set(int* flag, int slot) {
  nap();
  atomicExch(&flag[slot], 1);
}
__global__ void read_flags(int* flag, int* out, int slot) {
  out[slot] = atomicAdd(&flag[0], 0) * 100 + atomicAdd(&flag[1], 0) * 10 + atomicAdd(&flag[2], 0);
}
// A tail launch runs after the grid and everything it launched, whatever order they were launched in.
__global__ void tail_after_ff(int* f, int* o) {
  nap_set<<<1, 1, 0, cudaStreamFireAndForget>>>(f, 0);
  read_flags<<<1, 1, 0, cudaStreamTailLaunch>>>(f, o, 0);
}
__global__ void tail_before_ff(int* f, int* o) {
  read_flags<<<1, 1, 0, cudaStreamTailLaunch>>>(f, o, 1);
  nap_set<<<1, 1, 0, cudaStreamFireAndForget>>>(f, 0);
}
// ... and after the launching grid's own last instruction.
__global__ void tail_after_parent(int* f, int* o) {
  read_flags<<<1, 1, 0, cudaStreamTailLaunch>>>(f, o, 2);
  nap();
  atomicExch(&f[1], 1);
}
// A child's tail launch runs before its parent's.
__global__ void inner_tail(int* f) { nap_set<<<1, 1, 0, cudaStreamTailLaunch>>>(f, 2); }
__global__ void tail_of_child(int* f, int* o) {
  inner_tail<<<1, 1, 0, cudaStreamFireAndForget>>>(f);
  read_flags<<<1, 1, 0, cudaStreamTailLaunch>>>(f, o, 3);
}
// Tail launches run in order.
__global__ void two_tails(int* f, int* o) {
  nap_set<<<1, 1, 0, cudaStreamTailLaunch>>>(f, 0);
  read_flags<<<1, 1, 0, cudaStreamTailLaunch>>>(f, o, 4);
}
// The block's default stream, and a named one, keep their launches in order.
__global__ void default_stream(int* f, int* o) {
  nap_set<<<1, 1>>>(f, 0);
  read_flags<<<1, 1>>>(f, o, 5);
}
__global__ void named_stream(int* f, int* o) {
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  nap_set<<<1, 1, 0, s>>>(f, 0);
  read_flags<<<1, 1, 0, s>>>(f, o, 6);
  cudaStreamDestroy(s);
}
// An event orders one stream after another.
__global__ void event_order(int* f, int* o) {
  cudaStream_t s1, s2;
  cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
  cudaEvent_t e;
  cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  nap_set<<<1, 1, 0, s1>>>(f, 0);
  cudaEventRecord(e, s1);
  cudaStreamWaitEvent(s2, e, 0);
  read_flags<<<1, 1, 0, s2>>>(f, o, 7);
}
// A grid's tail launch waits for the grids its fire-and-forget children launched, and for the tails of
// a tail.
__global__ void ff_child(int* f) { nap_set<<<1, 1, 0, cudaStreamFireAndForget>>>(f, 0); }
__global__ void tail_after_grandchild(int* f, int* o) {
  ff_child<<<1, 1, 0, cudaStreamFireAndForget>>>(f);
  read_flags<<<1, 1, 0, cudaStreamTailLaunch>>>(f, o, 8);
}
__global__ void tail_of_tail(int* f, int* o) {
  ff_child<<<1, 1, 0, cudaStreamTailLaunch>>>(f);
  read_flags<<<1, 1, 0, cudaStreamTailLaunch>>>(f, o, 9);
}
// A tail launch waits for the grids in named streams too, launched before or after it.
__global__ void tail_after_named(int* f, int* o) {
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  nap_set<<<1, 1, 0, s>>>(f, 0);
  read_flags<<<1, 1, 0, cudaStreamTailLaunch>>>(f, o, 10);
  cudaStreamDestroy(s);
}
__global__ void tail_before_named(int* f, int* o) {
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  read_flags<<<1, 1, 0, cudaStreamTailLaunch>>>(f, o, 11);
  nap_set<<<1, 1, 0, s>>>(f, 0);
  cudaStreamDestroy(s);
}

bool run_streams() {
  typedef void (*TestKernel)(int*, int*);
  const struct {
    const char* name;
    TestKernel k;
    int want;
  } tests[] = {{"tail after a fire-and-forget grid", tail_after_ff, 100},
               {"tail launched before the fire-and-forget one", tail_before_ff, 100},
               {"tail after the parent's last instruction", tail_after_parent, 10},
               {"a child's tail before its parent's", tail_of_child, 1},
               {"two tails in order", two_tails, 100},
               {"the default stream in order", default_stream, 100},
               {"a named stream in order", named_stream, 100},
               {"an event between two streams", event_order, 100},
               {"tail after a grandchild", tail_after_grandchild, 100},
               {"tail after a tail's children", tail_of_tail, 100},
               {"tail after a named stream (launched before)", tail_after_named, 100},
               {"tail after a named stream (launched after)", tail_before_named, 100}};
  bool ok = true;
  int *flag, *out;
  CK(cudaMalloc(&flag, 3 * sizeof(int)));
  CK(cudaMalloc(&out, 16 * sizeof(int)));
  for (size_t i = 0; i < sizeof tests / sizeof *tests; ++i) {
    CK(cudaMemset(flag, 0, 3 * sizeof(int)));
    CK(cudaMemset(out, 0xff, 16 * sizeof(int)));
    tests[i].k<<<1, 1>>>(flag, out);
    CK(cudaDeviceSynchronize());
    int h[16];
    CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
    const bool good = h[i] == tests[i].want;
    std::printf("%s: flags %03d%s\n", tests[i].name, h[i], good ? "" : " (WRONG)");
    ok = ok && good;
  }
  CK(cudaFree(flag));
  CK(cudaFree(out));
  return ok;
}

// ---- 7. copies and fills from a kernel ------------------------------------------------------
__global__ void fill4(int* p, int v) {
  nap();
  for (int i = 0; i < 4; ++i) p[i] = v;
}
__global__ void sum4(const int* p, int* out) { *out = p[0] + p[1] + p[2] + p[3]; }
// A copy and a fill keep their place among a stream's launches.
__global__ void copy_order(int* a, int* b, int* out) {
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  fill4<<<1, 1, 0, s>>>(a, 7);
  out[2] = cudaMemcpyAsync(b, a, 16, cudaMemcpyDeviceToDevice, s);
  sum4<<<1, 1, 0, s>>>(b, out);
  out[3] = cudaMemsetAsync(b, 0, 16, s);
  sum4<<<1, 1, 0, s>>>(b, out + 1);
  cudaStreamDestroy(s);
}
__global__ void boxes(unsigned char* src, unsigned char* dst, int* err) {
  err[0] = cudaMemcpy2DAsync(dst, 8, src, 16, 4, 3, cudaMemcpyDeviceToDevice, 0);
  cudaMemcpy3DParms p;
  memset(&p, 0, sizeof p);
  p.srcPtr.ptr = src;
  p.srcPtr.pitch = 16;
  p.srcPtr.xsize = 16;
  p.srcPtr.ysize = 4;
  p.dstPtr.ptr = dst + 64;
  p.dstPtr.pitch = 16;
  p.dstPtr.xsize = 16;
  p.dstPtr.ysize = 4;
  p.extent.width = 4;
  p.extent.height = 2;
  p.extent.depth = 2;
  p.kind = cudaMemcpyDeviceToDevice;
  err[1] = cudaMemcpy3DAsync(&p, 0);
  err[2] = cudaMemset2DAsync(dst + 256, 8, 0, 4, 3, 0);
  cudaPitchedPtr pp;
  pp.ptr = dst + 512;
  pp.pitch = 16;
  pp.xsize = 16;
  pp.ysize = 4;
  cudaExtent ext;
  ext.width = 4;
  ext.height = 2;
  ext.depth = 2;
  err[3] = cudaMemset3DAsync(pp, 0, ext, 0);
}
__global__ void copy_errors(int* a, int* b, int* out) {
  int n = 0;
  for (int k = -1; k <= 5; ++k) out[n++] = cudaMemcpyAsync(b, a, 16, (cudaMemcpyKind)k, 0);
  out[n++] = cudaMemcpyAsync(b, a, 0, cudaMemcpyDeviceToDevice, 0);
  out[n++] = cudaMemsetAsync(b, 0, 0, 0);
  out[n++] = cudaMemcpyAsync(b, a, 16, cudaMemcpyDeviceToDevice, (cudaStream_t)0x1234);
  out[n++] = cudaMemsetAsync(b, 0, 16, (cudaStream_t)0x1234);
  out[n++] = cudaMemcpyAsync(b, a, 16, cudaMemcpyDeviceToDevice, cudaStreamTailLaunch);
  out[n++] = cudaMemcpyAsync(nullptr, a, 16, cudaMemcpyDeviceToDevice, 0);
  out[n++] = cudaMemcpyAsync(b, nullptr, 16, cudaMemcpyDeviceToDevice, 0);
  out[n++] = cudaMemsetAsync(nullptr, 0, 16, 0);
  out[n++] = cudaMemcpy2DAsync(b, 2, a, 16, 4, 3, cudaMemcpyDeviceToDevice, 0);
  out[n++] = cudaMemcpy2DAsync(b, 8, a, 2, 4, 3, cudaMemcpyDeviceToDevice, 0);
  out[n++] = cudaMemset2DAsync(b, 2, 0, 4, 3, 0);
}

bool run_copies() {
  bool ok = true;
  {
    int *a, *b, *out;
    CK(cudaMalloc(&a, 64));
    CK(cudaMalloc(&b, 64));
    CK(cudaMalloc(&out, 16));
    CK(cudaMemset(a, 0, 64));
    CK(cudaMemset(b, 0, 64));
    copy_order<<<1, 1>>>(a, b, out);
    CK(cudaDeviceSynchronize());
    int h[4];
    CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
    const bool good = h[0] == 28 && h[1] == 0 && h[2] == 0 && h[3] == 0;
    std::printf("copy and fill in stream order: %d then %d (calls %d %d)%s\n", h[0], h[1], h[2], h[3],
                good ? "" : " (want 28 then 0, calls 0 0)");
    ok = ok && good;
    CK(cudaFree(a));
    CK(cudaFree(b));
    CK(cudaFree(out));
  }
  {
    unsigned char src[256], dst[1024], want[1024];
    for (int i = 0; i < 256; ++i) src[i] = static_cast<unsigned char>(i);
    std::memset(dst, 0xEE, sizeof dst);
    std::memcpy(want, dst, sizeof want);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c) {
        want[r * 8 + c] = src[r * 16 + c];   // the 2D copy: rows of 4, pitch 16 to pitch 8
        want[256 + r * 8 + c] = 0;           // the 2D fill
      }
    for (int z = 0; z < 2; ++z)
      for (int y = 0; y < 2; ++y)
        for (int c = 0; c < 4; ++c) {
          want[64 + z * 64 + y * 16 + c] = src[z * 64 + y * 16 + c];   // the 3D copy
          want[512 + z * 64 + y * 16 + c] = 0;                        // the 3D fill
        }
    unsigned char *ds, *dd;
    int* err;
    CK(cudaMalloc(&ds, sizeof src));
    CK(cudaMalloc(&dd, sizeof dst));
    CK(cudaMalloc(&err, 16));
    CK(cudaMemcpy(ds, src, sizeof src, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd, dst, sizeof dst, cudaMemcpyHostToDevice));
    boxes<<<1, 1>>>(ds, dd, err);
    CK(cudaDeviceSynchronize());
    int e[4];
    CK(cudaMemcpy(e, err, sizeof e, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dst, dd, sizeof dst, cudaMemcpyDeviceToHost));
    const bool good = std::memcmp(dst, want, sizeof dst) == 0 && !e[0] && !e[1] && !e[2] && !e[3];
    std::printf("2D and 3D copies and fills: %s\n", good ? "as expected" : "WRONG");
    ok = ok && good;
    CK(cudaFree(ds));
    CK(cudaFree(dd));
    CK(cudaFree(err));
  }
  {
    int *a, *b, *out;
    CK(cudaMalloc(&a, 64));
    CK(cudaMalloc(&b, 64));
    CK(cudaMalloc(&out, 64 * sizeof(int)));
    copy_errors<<<1, 1>>>(a, b, out);
    CK(cudaDeviceSynchronize());
    int h[64];
    CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
    // kinds -1 to 5 (only device to device and default are allowed: 21 is cudaErrorInvalidMemcpyDirection),
    // empty copy and fill, a stream that is none (999: cudaErrorUnknown), a tail stream, null pointers (1),
    // pitches too small for the width (12: cudaErrorInvalidPitchValue; the fill's is 1).
    const int want[] = {21, 21, 21, 21, 0, 0, 21, 0, 0, 999, 999, 0, 1, 1, 1, 12, 12, 1};
    bool good = true;
    for (size_t i = 0; i < sizeof want / sizeof *want; ++i) good = good && h[i] == want[i];
    std::printf("copy and fill errors:");
    for (size_t i = 0; i < sizeof want / sizeof *want; ++i) std::printf(" %d", h[i]);
    std::printf("%s\n", good ? "" : " (WRONG)");
    ok = ok && good;
    CK(cudaFree(a));
    CK(cudaFree(b));
    CK(cudaFree(out));
  }
  return ok;
}

// ---- 8. cudaMalloc and cudaFree from a kernel ------------------------------------------------
__global__ void heap_alloc_k(int* out, void** keep) {
  int* p = nullptr;
  out[0] = cudaMalloc((void**)&p, 64);
  out[1] = p != nullptr;
  p[0] = 1234;
  *keep = p;
}
__global__ void heap_use_k(int* out, void** keep) {
  int* p = static_cast<int*>(*keep);
  out[2] = p[0];
  out[3] = cudaFree(p);
}
__global__ void heap_errors(int* out) {
  void* p = reinterpret_cast<void*>(1);
  out[0] = cudaMalloc(&p, 0);
  out[1] = reinterpret_cast<size_t>(p) == 1;
  out[2] = cudaMalloc(&p, static_cast<size_t>(1) << 40);
  out[3] = reinterpret_cast<size_t>(p) == 1;
  out[4] = cudaFree(nullptr);
  int local = 0;
  out[5] = cudaFree(&local);
  void* q = nullptr;
  out[6] = cudaMalloc(&q, 100);
  out[7] = cudaFree(q);
  void* r = malloc(32);
  out[8] = cudaFree(r);
  out[9] = cudaGetLastError();
}

bool run_heap() {
  bool ok = true;
  int *out, h[10];
  void** keep;
  CK(cudaMalloc(&out, sizeof h));
  CK(cudaMalloc(&keep, sizeof(void*)));
  CK(cudaMemset(out, 0xff, sizeof h));
  heap_alloc_k<<<1, 1>>>(out, keep);
  CK(cudaDeviceSynchronize());
  heap_use_k<<<1, 1>>>(out, keep);   // memory a kernel allocated is there for the next
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, out, 4 * sizeof(int), cudaMemcpyDeviceToHost));
  bool good = h[0] == 0 && h[1] == 1 && h[2] == 1234 && h[3] == 0;
  std::printf("cudaMalloc in one kernel, cudaFree in the next: %d %d %d %d%s\n", h[0], h[1], h[2], h[3],
              good ? "" : " (want 0 1 1234 0)");
  ok = ok && good;
  heap_errors<<<1, 1>>>(out);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
  // 0 bytes: cudaErrorInvalidValue, the pointer left alone; far too much: cudaErrorMemoryAllocation (2), the
  // pointer left alone; freeing null, a local variable, a block cudaMalloc made or malloc made: no error.
  const int want[] = {1, 1, 2, 1, 0, 0, 0, 0, 0, 2};
  good = true;
  for (int i = 0; i < 10; ++i) good = good && h[i] == want[i];
  std::printf("cudaMalloc and cudaFree errors:");
  for (int i = 0; i < 10; ++i) std::printf(" %d", h[i]);
  std::printf("%s\n", good ? "" : " (WRONG)");
  ok = ok && good;
  CK(cudaFree(out));
  CK(cudaFree(keep));
  return ok;
}

// ---- 9. what the host's CUDA library knows, asked from a kernel ----------------------------------
__global__ void attr_plain() {}
__global__ void attr_shared() {
  __shared__ char s[4096];
  s[threadIdx.x] = 1;
  __syncthreads();
  if (s[1] == 5) printf("x");
}
__global__ void attr_local(int n) {
  volatile int a[64];
  for (int i = 0; i < 64; ++i) a[i] = i * n;
  int t = 0;
  for (int i = 0; i < 64; ++i) t += a[(i * n) & 63];
  if (t == -1) printf("x");
}
// cudaFuncGetAttributes fills the first seven fields (sizes, thread bound, registers, versions).
__global__ void get_func_attrs(long long* out) {
  const void* fs[3] = {reinterpret_cast<const void*>(attr_plain), reinterpret_cast<const void*>(attr_shared),
                       reinterpret_cast<const void*>(attr_local)};
  for (int i = 0; i < 3; ++i) {
    cudaFuncAttributes a;
    memset(&a, 0xAB, sizeof a);
    out[i * 8] = cudaFuncGetAttributes(&a, fs[i]);
    out[i * 8 + 1] = static_cast<long long>(a.sharedSizeBytes);
    out[i * 8 + 2] = static_cast<long long>(a.constSizeBytes);
    out[i * 8 + 3] = static_cast<long long>(a.localSizeBytes);
    out[i * 8 + 4] = a.maxThreadsPerBlock;
    out[i * 8 + 5] = a.numRegs;
    out[i * 8 + 6] = a.ptxVersion;
    out[i * 8 + 7] = a.binaryVersion;
  }
}
// The attribute numbers 1 to 120 (the toolkits this builds with name them all).
__global__ void get_attrs(int* err, int* val) {
  for (int a = 1; a <= 120; ++a) {
    int v = -12345;
    err[a] = cudaDeviceGetAttribute(&v, static_cast<cudaDeviceAttr>(a), 0);
    val[a] = v;
  }
}
__global__ void get_attr_errors(int* out) {
  int v = 77;
  out[0] = cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 5);   // no such device: cudaErrorInvalidDevice
  out[1] = v;                                                               // left alone
  out[2] = cudaDeviceGetAttribute(&v, static_cast<cudaDeviceAttr>(9999), 0);   // cudaErrorInvalidValue
  out[3] = v;
}
__global__ void get_limits(unsigned long long* out) {
  const cudaLimit ls[7] = {cudaLimitStackSize,
                           cudaLimitPrintfFifoSize,
                           cudaLimitMallocHeapSize,
                           cudaLimitDevRuntimeSyncDepth,
                           cudaLimitDevRuntimePendingLaunchCount,
                           cudaLimitMaxL2FetchGranularity,
                           cudaLimitPersistingL2CacheSize};
  for (int i = 0; i < 7; ++i) {
    size_t v = 0;
    out[2 * i] = cudaDeviceGetLimit(&v, ls[i]);
    out[2 * i + 1] = v;
  }
}
__global__ void get_config(int* out) {
  cudaFuncCache c = static_cast<cudaFuncCache>(77);
  out[0] = cudaDeviceGetCacheConfig(&c);
  out[1] = static_cast<int>(c);
  cudaSharedMemConfig m = static_cast<cudaSharedMemConfig>(77);
  out[2] = cudaDeviceGetSharedMemConfig(&m);   // deprecated, and still answered
  out[3] = static_cast<int>(m);
  int v = -1;
  out[4] = cudaRuntimeGetVersion(&v);
  out[5] = v;
}
// Occupancy: the blocks of a kernel one multiprocessor holds.
__global__ void get_occupancy(int* out) {
  const void* fs[3] = {reinterpret_cast<const void*>(attr_plain), reinterpret_cast<const void*>(attr_shared),
                       reinterpret_cast<const void*>(attr_local)};
  const int sizes[6] = {32, 128, 256, 1024, 2048, 0};
  const size_t dyn[3] = {0, 4096, 40000};
  int n = 0;
  for (int f = 0; f < 3; ++f)
    for (int b = 0; b < 6; ++b)
      for (int d = 0; d < 3; ++d) {
        int blocks = -99;
        out[n++] = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fs[f], sizes[b], dyn[d]);
        out[n++] = blocks;
        blocks = -99;
        out[n++] = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&blocks, fs[f], sizes[b], dyn[d], 0);
        out[n++] = blocks;
      }
  int min_grid = -1, block = -1;
  out[n++] = cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, attr_plain, 0, 0);
  out[n++] = min_grid;
  out[n++] = block;
  out[n++] = cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, attr_shared, 0, 0);
  out[n++] = min_grid;
  out[n++] = block;
}
// Error names and strings: the text in device memory, as the host's.
constexpr int kStrCodes = 48;
__global__ void get_strings(char* names, char* texts) {
  for (int i = 0; i < kStrCodes; ++i) {
    // 0 to 39 and a few well past what is named.
    const int code = i < 40 ? i : (i == 40 ? 700 : (i == 41 ? 719 : (i == 42 ? 999 : (i == 43 ? 12345 : i))));
    const char* n = cudaGetErrorName(static_cast<cudaError_t>(code));
    const char* t = cudaGetErrorString(static_cast<cudaError_t>(code));
    int j = 0;
    for (; j < 63 && n[j]; ++j) names[i * 64 + j] = n[j];
    names[i * 64 + j] = 0;
    j = 0;
    for (; j < 63 && t[j]; ++j) texts[i * 64 + j] = t[j];
    texts[i * 64 + j] = 0;
  }
}

bool run_queries() {
  bool ok = true;
  const void* fs[3] = {reinterpret_cast<const void*>(attr_plain), reinterpret_cast<const void*>(attr_shared),
                       reinterpret_cast<const void*>(attr_local)};
  {
    long long* d;
    long long h[24];
    CK(cudaMalloc(&d, sizeof h));
    get_func_attrs<<<1, 1>>>(d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    bool good = true;
    for (int i = 0; i < 3; ++i) {
      cudaFuncAttributes a;
      CK(cudaFuncGetAttributes(&a, fs[i]));
      const long long want[8] = {0,
                                 static_cast<long long>(a.sharedSizeBytes),
                                 static_cast<long long>(a.constSizeBytes),
                                 static_cast<long long>(a.localSizeBytes),
                                 a.maxThreadsPerBlock,
                                 a.numRegs,
                                 a.ptxVersion,
                                 a.binaryVersion};
      for (int k = 0; k < 8; ++k) good = good && h[i * 8 + k] == want[k];
    }
    good = good && h[8 + 1] == 4096;   // attr_shared's static shared memory
    std::printf("cudaFuncGetAttributes in a kernel, as the host's: %s\n", good ? "same" : "DIFFERENT");
    ok = ok && good;
    CK(cudaFree(d));
  }
  {
    int *err, *val;
    CK(cudaMalloc(&err, 128 * sizeof(int)));
    CK(cudaMalloc(&val, 128 * sizeof(int)));
    get_attrs<<<1, 1>>>(err, val);
    CK(cudaDeviceSynchronize());
    int he[128], hv[128];
    CK(cudaMemcpy(he, err, sizeof he, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hv, val, sizeof hv, cudaMemcpyDeviceToHost));
    int differ = 0, agreed = 0;
    for (int a = 1; a <= 120; ++a) {
      int v = -12345;
      const cudaError_t e = cudaDeviceGetAttribute(&v, static_cast<cudaDeviceAttr>(a), 0);
      cudaGetLastError();
      if (static_cast<int>(e) != he[a] || (e == cudaSuccess && v != hv[a])) ++differ;
      if (e == cudaSuccess) ++agreed;
    }
    const bool good = differ == 0 && agreed > 20;
    std::printf("cudaDeviceGetAttribute in a kernel, attributes 1 to 120, as the host's: %s\n",
                good ? "same" : "DIFFERENT");
    ok = ok && good;
    int* eo;
    CK(cudaMalloc(&eo, 4 * sizeof(int)));
    get_attr_errors<<<1, 1>>>(eo);
    CK(cudaDeviceSynchronize());
    int h[4];
    CK(cudaMemcpy(h, eo, sizeof h, cudaMemcpyDeviceToHost));
    const bool good2 = h[0] == cudaErrorInvalidDevice && h[1] == 77 && h[2] == cudaErrorInvalidValue && h[3] == 77;
    std::printf("cudaDeviceGetAttribute errors: %d %d %d %d%s\n", h[0], h[1], h[2], h[3],
                good2 ? "" : " (want 101 77 1 77)");
    ok = ok && good2;
    CK(cudaFree(err));
    CK(cudaFree(val));
    CK(cudaFree(eo));
  }
  {
    // The limits a kernel reads are the host's current ones: after the host sets one it reads back.
    const cudaLimit ls[7] = {cudaLimitStackSize,
                             cudaLimitPrintfFifoSize,
                             cudaLimitMallocHeapSize,
                             cudaLimitDevRuntimeSyncDepth,
                             cudaLimitDevRuntimePendingLaunchCount,
                             cudaLimitMaxL2FetchGranularity,
                             cudaLimitPersistingL2CacheSize};
    CK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 64));
    unsigned long long* d;
    unsigned long long h[14];
    CK(cudaMalloc(&d, sizeof h));
    get_limits<<<1, 1>>>(d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    bool good = true;
    for (int i = 0; i < 7; ++i) {
      size_t v = 0;
      const cudaError_t e = cudaDeviceGetLimit(&v, ls[i]);
      cudaGetLastError();
      good = good && h[2 * i] == static_cast<unsigned long long>(e) && (e != cudaSuccess || h[2 * i + 1] == v);
    }
    good = good && h[8] == 0 && h[9] == 64;   // the limit just set
    std::printf("cudaDeviceGetLimit in a kernel, as the host's (pending launches %llu): %s\n", h[9],
                good ? "same" : "DIFFERENT");
    ok = ok && good;
    CK(cudaFree(d));
  }
  {
    int* d;
    int h[8];
    CK(cudaMalloc(&d, sizeof h));
    get_config<<<1, 1>>>(d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    cudaFuncCache cc;
    CK(cudaDeviceGetCacheConfig(&cc));
    // The device runtime's own version is 6000 whatever the toolkit; the shared-memory banks are four bytes.
    const bool good = h[0] == 0 && h[1] == static_cast<int>(cc) && h[2] == 0 && h[3] == 1 && h[4] == 0 && h[5] == 6000;
    std::printf("cache configuration %d, shared memory bank %d, runtime version %d%s\n", h[1], h[3], h[5],
                good ? "" : " (WRONG)");
    ok = ok && good;
    CK(cudaFree(d));
  }
  {
    int* d;
    constexpr int kN = 3 * 6 * 3 * 4 + 6;
    int h[kN];
    CK(cudaMalloc(&d, sizeof h));
    get_occupancy<<<1, 1>>>(d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    const int sizes[6] = {32, 128, 256, 1024, 2048, 0};
    const size_t dyn[3] = {0, 4096, 40000};
    int n = 0, differ = 0;
    for (int f = 0; f < 3; ++f)
      for (int b = 0; b < 6; ++b)
        for (int dd = 0; dd < 3; ++dd) {
          int blocks = -99;
          const cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fs[f], sizes[b], dyn[dd]);
          cudaGetLastError();
          // A block size of 0 is refused by both (the device runtime writes 0 where the host leaves the count).
          if (sizes[b] == 0) {
            differ += static_cast<int>(e) != h[n] || h[n] != cudaErrorInvalidValue;
            differ += static_cast<int>(e) != h[n + 2];
          } else {
            differ += static_cast<int>(e) != h[n] || blocks != h[n + 1];
            differ += static_cast<int>(e) != h[n + 2] || blocks != h[n + 3];
          }
          n += 4;
        }
    int min_grid = -1, block = -1;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, attr_plain, 0, 0);
    differ += h[n] != 0 || h[n + 1] != min_grid || h[n + 2] != block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, attr_shared, 0, 0);
    differ += h[n + 3] != 0 || h[n + 4] != min_grid || h[n + 5] != block;
    const bool good = differ == 0;
    std::printf("occupancy in a kernel, as the host's: %s\n", good ? "same" : "DIFFERENT");
    ok = ok && good;
    CK(cudaFree(d));
  }
  {
    char *names, *texts;
    CK(cudaMalloc(&names, kStrCodes * 64));
    CK(cudaMalloc(&texts, kStrCodes * 64));
    get_strings<<<1, 1>>>(names, texts);
    CK(cudaDeviceSynchronize());
    std::vector<char> hn(kStrCodes * 64), ht(kStrCodes * 64);
    CK(cudaMemcpy(hn.data(), names, hn.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ht.data(), texts, ht.size(), cudaMemcpyDeviceToHost));
    int differ = 0;
    for (int i = 0; i < kStrCodes; ++i) {
      const int code = i < 40 ? i : (i == 40 ? 700 : (i == 41 ? 719 : (i == 42 ? 999 : (i == 43 ? 12345 : i))));
      const cudaError_t e = static_cast<cudaError_t>(code);
      differ += std::strncmp(&hn[i * 64], cudaGetErrorName(e), 63) != 0;
      differ += std::strncmp(&ht[i * 64], cudaGetErrorString(e), 63) != 0;
    }
    const bool good = differ == 0;
    std::printf("cudaGetErrorName and cudaGetErrorString in a kernel, as the host's: %s\n", good ? "same" : "DIFFERENT");
    ok = ok && good;
    CK(cudaFree(names));
    CK(cudaFree(texts));
  }
  return ok;
}

// ---- 10. streams and events: what their calls accept and refuse ----------------------------------
__global__ void noop2() {}
__global__ void stream_calls(int* r) {
  int n = 0;
  cudaStream_t s;
  r[n++] = cudaStreamCreateWithFlags(&s, cudaStreamDefault);          // 0: any of 0 and cudaStreamNonBlocking
  r[n++] = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);      // 0
  cudaStream_t bad;
  r[n++] = cudaStreamCreateWithFlags(&bad, 2);                        // 1
  r[n++] = cudaStreamCreateWithFlags(&bad, 7);                        // 1
  r[n++] = cudaStreamCreateWithFlags(nullptr, cudaStreamNonBlocking); // 1
  r[n++] = cudaStreamDestroy(s);                                      // 0
  r[n++] = cudaStreamDestroy(s);                                      // 0: no error twice
  r[n++] = cudaStreamDestroy(0);                                      // 1: the default stream is not destroyed
  r[n++] = cudaStreamDestroy(cudaStreamPerThread);                    // 1
  r[n++] = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(0x1234)); // 1
  for (unsigned f = 0; f < 16; ++f) {
    cudaEvent_t e;
    r[n++] = cudaEventCreateWithFlags(&e, f);   // 0 only for 2, 3, 10, 11: no timing is required
  }
  cudaEvent_t ev;
  cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  cudaStream_t s2;
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
  r[n++] = cudaEventRecord(ev, s2);                                        // 0
  r[n++] = cudaEventRecord(ev, 0);                                         // 0
  r[n++] = cudaEventRecord(ev, reinterpret_cast<cudaStream_t>(0x1234));    // 1
  r[n++] = cudaStreamWaitEvent(s2, ev, 0);                                 // 0
  r[n++] = cudaStreamWaitEvent(0, ev, 0);                                  // 0
  r[n++] = cudaEventDestroy(ev);                                           // 0
  r[n++] = cudaEventDestroy(ev);                                           // 0: and again
  cudaGetLastError();
  // Launches: the streams that are streams, and ones that are not.
  const unsigned long long ids[] = {0, 1, 2, 3, 4, 5, 6, 8, 0x10, 0x100, 0x1000, 0x1234, 0x0100000000000000ull,
                                    0x0300000000000000ull, ~0ull};
  for (int i = 0; i < 15; ++i) {
    noop2<<<1, 1, 0, reinterpret_cast<cudaStream_t>(ids[i])>>>();
    r[n++] = cudaGetLastError();   // 0 for 0 to 4, 1 for the rest
  }
  noop2<<<1, 1, 0, s2>>>();
  r[n++] = cudaGetLastError();     // 0: a stream of this block's
  cudaStreamDestroy(s2);
  noop2<<<1, 1, 0, s2>>>();
  r[n++] = cudaGetLastError();     // 0: the card does not remember it is gone
}

bool run_streams_api() {
  int* d;
  int h[64];
  CK(cudaMalloc(&d, sizeof h));
  CK(cudaMemset(d, 0xff, sizeof h));
  stream_calls<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  std::vector<int> want = {0, 0, 1, 1, 1, 0, 0, 1, 1, 1};
  const int events[16] = {1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1};
  want.insert(want.end(), events, events + 16);
  const int tail[] = {0, 0, 1, 0, 0, 0, 0};
  want.insert(want.end(), tail, tail + 7);
  const int launches[15] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  want.insert(want.end(), launches, launches + 15);
  want.push_back(0);
  want.push_back(0);
  bool good = true;
  std::printf("streams and events:");
  for (size_t i = 0; i < want.size(); ++i) {
    good = good && h[i] == want[i];
    std::printf(" %d", h[i]);
  }
  std::printf("%s\n", good ? "" : " (WRONG)");
  CK(cudaFree(d));
  return good;
}

// ---- 11. launch configurations a child may not have --------------------------------------------
__global__ void bounded64() __launch_bounds__(64);
__global__ void bounded64() {}
__global__ void config_calls(int* r) {
  int n = 0;
  noop2<<<0, 1>>>();
  r[n++] = cudaGetLastError();          // 9: cudaErrorInvalidConfiguration for a zero dimension,
  noop2<<<1, 0>>>();
  r[n++] = cudaGetLastError();
  noop2<<<1, 1025>>>();                 // too many threads,
  r[n++] = cudaGetLastError();
  noop2<<<1, dim3(1, 1, 65)>>>();       // a block dimension past the limit,
  r[n++] = cudaGetLastError();
  noop2<<<dim3(1, 65536, 1), 1>>>();    // a grid dimension past it,
  r[n++] = cudaGetLastError();
  noop2<<<1, 1, 1 << 20>>>();           // too much shared memory,
  r[n++] = cudaGetLastError();
  bounded64<<<1, 65>>>();               // a block past __launch_bounds__.
  r[n++] = cudaGetLastError();
  bounded64<<<1, 64>>>();               // within it,
  r[n++] = cudaGetLastError();
  noop2<<<dim3(1, 1, 65), 1>>>();       // and a grid z of 65, which is allowed.
  r[n++] = cudaGetLastError();
}

bool run_config() {
  int* d;
  int h[16];
  CK(cudaMalloc(&d, sizeof h));
  config_calls<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  const int want[] = {9, 9, 9, 9, 9, 9, 9, 0, 0};
  bool good = true;
  std::printf("launch configurations:");
  for (size_t i = 0; i < sizeof want / sizeof *want; ++i) {
    good = good && h[i] == want[i];
    std::printf(" %d", h[i]);
  }
  std::printf("%s\n", good ? "" : " (WRONG)");
  CK(cudaFree(d));
  return good;
}

// ---- 12. the pending-launch limit ---------------------------------------------------------------
// A grid is not complete until what it launched is, so a chain of grids each launching the next
// has one more incomplete grid at every level: its launch is refused at the limit (an RTX 3060:
// the 33rd with a limit below 32, otherwise at the limit's own number plus one).
__global__ void chain(int* out, int d, int stop) {
  atomicMax(&out[0], d);
  if (d >= stop) return;
  chain<<<1, 1>>>(out, d + 1, stop);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    out[1] = d;
    out[2] = e;
  }
}
__global__ void tick(int* c) { atomicAdd(c, 1); }
// More launches than the limit, each into its own fire-and-forget stream, run: the ones issued
// earlier have finished by the time the limit is reached (or, on the card, are finishing).
__global__ void flood(int* c, int* out, int n) {
  int failed = 0;
  for (int i = 0; i < n; ++i) {
    tick<<<1, 1, 0, cudaStreamFireAndForget>>>(c);
    if (cudaGetLastError() != cudaSuccess) ++failed;
  }
  out[0] = failed;
}

bool run_pending() {
  bool ok = true;
  int* d;
  CK(cudaMalloc(&d, 16 * sizeof(int)));
  const struct {
    int limit, stop, reached, failed_at, err;
  } chains[] = {{64, 1000, 65, 65, 69},   // the 65th grid's launch fails
                {10, 1000, 33, 33, 69},   // a limit below 32 is 32
                {2048, 100, 100, 0, 0},   // nesting deeper than 24 is fine
                {2048, 3, 3, 0, 0}};
  for (const auto& c : chains) {
    CK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, c.limit));
    CK(cudaMemset(d, 0, 16 * sizeof(int)));
    chain<<<1, 1>>>(d, 1, c.stop);
    CK(cudaDeviceSynchronize());
    int h[4];
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    const bool good = h[0] == c.reached && h[1] == c.failed_at && h[2] == c.err;
    std::printf("a chain of launches, limit %d: reached %d, first launch refused at %d with %d%s\n", c.limit, h[0],
                h[1], h[2], good ? "" : " (WRONG)");
    ok = ok && good;
  }
  CK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 2048));
  {
    const int n = 3000;
    int *count, *failed;
    CK(cudaMalloc(&count, sizeof(int)));
    CK(cudaMalloc(&failed, sizeof(int)));
    CK(cudaMemset(count, 0, sizeof(int)));
    CK(cudaMemset(failed, 0, sizeof(int)));
    flood<<<1, 1>>>(count, failed, n);
    CK(cudaDeviceSynchronize());
    int hc = 0, hf = 0;
    CK(cudaMemcpy(&hc, count, sizeof hc, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hf, failed, sizeof hf, cudaMemcpyDeviceToHost));
    const bool good = hc == n - hf && hf >= 0 && hf < n;
    std::printf("%d launches, each counted once if it ran and refused if not: %s\n", n, good ? "consistent" : "WRONG");
    ok = ok && good;
    CK(cudaFree(count));
    CK(cudaFree(failed));
  }
  CK(cudaFree(d));
  return ok;
}

// ---- 13. the older pair: cudaGetParameterBuffer and cudaLaunchDevice -----------------------------
__global__ void params_k(int a, float b, long long* out) {
  out[0] = a;
  out[1] = static_cast<long long>(b * 4);
}
__global__ void old_api(long long* out) {
  void* buf = cudaGetParameterBuffer(8, 16);   // int at 0, float at 4, a pointer at 8: the kernel's own layout
  *reinterpret_cast<int*>(buf) = 41;
  *reinterpret_cast<float*>(static_cast<char*>(buf) + 4) = 2.5f;
  *reinterpret_cast<long long**>(static_cast<char*>(buf) + 8) = out;
  out[2] = cudaLaunchDevice(reinterpret_cast<void*>(params_k), buf, dim3(1), dim3(1), 0, 0);
  out[3] = cudaLaunchDevice(reinterpret_cast<void*>(noop2), nullptr, dim3(1), dim3(1), 0, 0);   // no parameters
  void* big = cudaGetParameterBuffer(8, 1 << 20);                                              // any size
  out[4] = big != nullptr;
  out[5] = cudaLaunchDevice(reinterpret_cast<void*>(noop2), nullptr, dim3(0), dim3(1), 0, 0);   // refused
}

bool run_old_api() {
  long long* d;
  long long h[6];
  CK(cudaMalloc(&d, sizeof h));
  CK(cudaMemset(d, 0, sizeof h));
  old_api<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  const bool good = h[0] == 41 && h[1] == 10 && h[2] == 0 && h[3] == 0 && h[4] == 1 && h[5] == 9;
  std::printf("cudaGetParameterBuffer and cudaLaunchDevice: %lld %lld, calls %lld %lld %lld %lld%s\n", h[0], h[1], h[2],
              h[3], h[4], h[5], good ? "" : " (want 41 10, calls 0 0 1 9)");
  CK(cudaFree(d));
  return good;
}

// ---- 14. the last error is each thread's own ------------------------------------------------
__global__ void last_error(int* out) {
  if (threadIdx.x == 0) noop2<<<1, 1025>>>();   // thread 0's launch fails
  __syncthreads();
  out[threadIdx.x] = cudaPeekAtLastError();
  __syncthreads();
  out[32 + threadIdx.x] = cudaGetLastError();
  __syncthreads();
  out[64 + threadIdx.x] = cudaGetLastError();
}

bool run_last_error() {
  int* d;
  int h[96];
  CK(cudaMalloc(&d, sizeof h));
  CK(cudaMemset(d, 0xff, sizeof h));
  last_error<<<1, 4>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  bool good = true;
  for (int t = 0; t < 4; ++t) {
    good = good && h[t] == (t == 0 ? 9 : 0) && h[32 + t] == (t == 0 ? 9 : 0) && h[64 + t] == 0;
  }
  std::printf("last error of thread 0 and the others: %d %d %d %d, then %d %d %d %d, then %d%s\n", h[0], h[1], h[2], h[3],
              h[32], h[33], h[34], h[35], h[64], good ? "" : " (WRONG)");
  CK(cudaFree(d));
  return good;
}

bool run() {
  bool ok = true;
  {
    int* d;
    const int max_depth = 3;
    CK(cudaMalloc(&d, (max_depth + 1) * sizeof(int)));
    CK(cudaMemset(d, 0, (max_depth + 1) * sizeof(int)));
    levels<<<2, 2>>>(d, 0, max_depth);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    int h[max_depth + 1];
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    // 2 blocks at level 0; each of their 4 threads launches 2 blocks. The
    // deepest level is 16 grids, each with a block 0 (one of them prints).
    const bool good = h[0] == 2 && h[1] == 8 && h[2] == 32 && h[3] == 16;
    std::printf("recursive launches: %d %d %d, %d grids at the bottom%s\n", h[0], h[1], h[2], h[3],
                good ? "" : " (want 2 8 32, 16)");
    ok = ok && good;
    CK(cudaFree(d));
  }
  {
    const int n = 512;
    unsigned h[n];
    unsigned x = 12345;
    for (int i = 0; i < n; ++i) {
      x = x * 1103515245u + 12345u;
      h[i] = (x >> 8) % 1000;
    }
    unsigned* d;
    CK(cudaMalloc(&d, sizeof h));
    CK(cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice));
    quicksort<<<1, 1>>>(d, 0, n - 1, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    unsigned long long sum = 0;
    bool sorted = true;
    for (int i = 0; i < n; ++i) {
      sum += h[i];
      if (i && h[i - 1] > h[i]) sorted = false;
    }
    std::printf("quicksort in device streams: %s, sum %llu\n", sorted ? "sorted" : "NOT SORTED", sum);
    ok = ok && sorted;
    CK(cudaFree(d));
  }
  {
    int *in, *out;
    int h[256], s[4];
    for (int i = 0; i < 256; ++i) h[i] = i;
    CK(cudaMalloc(&in, sizeof h));
    CK(cudaMalloc(&out, sizeof s));
    CK(cudaMemcpy(in, h, sizeof h, cudaMemcpyHostToDevice));
    launch_sums<<<1, 1>>>(in, out);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(s, out, sizeof s, cudaMemcpyDeviceToHost));
    bool good = true;
    for (int b = 0; b < 4; ++b) good = good && s[b] == 64 * 64 * b + 64 * 63 / 2;
    std::printf("child dynamic shared memory: %d %d %d %d%s\n", s[0], s[1], s[2], s[3], good ? "" : " (want 2016 6112 10208 14304)");
    ok = ok && good;
    CK(cudaFree(in));
    CK(cudaFree(out));
  }
  {
    int* d;
    int h[8];
    CK(cudaMalloc(&d, sizeof h));
    CK(cudaMemset(d, 0xff, sizeof h));
    errors<<<1, 1>>>(d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    int dev = -1, count = -1;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceCount(&count));
    const bool good = h[0] == cudaSuccess && h[1] == cudaErrorInvalidConfiguration &&
                      h[2] == cudaErrorInvalidConfiguration && h[3] == cudaSuccess;
    std::printf("device last error: %d %d %d %d%s\n", h[0], h[1], h[2], h[3], good ? "" : " (want 0 9 9 0)");
    const bool asked = h[4] == cudaSuccess && h[5] == dev && h[6] == cudaSuccess && h[7] == count;
    std::printf("device-side cudaGetDevice: %s\n", asked ? "as the host's" : "WRONG");
    ok = ok && good && asked;
    CK(cudaFree(d));
  }
  ok = run_streams() && ok;
  ok = run_copies() && ok;
  ok = run_heap() && ok;
  ok = run_queries() && ok;
  ok = run_streams_api() && ok;
  ok = run_config() && ok;
  ok = run_pending() && ok;
  ok = run_old_api() && ok;
  ok = run_last_error() && ok;
  return ok;
}

int main() {
  const bool ok = run();
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
