// Device memory shared between two processes, which is what CUDA IPC is for:
// one process allocates and fills a buffer, hands its handle to another, and
// the second one's kernel reads and writes the same bytes. vLLM shares a KV
// cache this way; NCCL uses it for peer buffers on one machine.
//
//   ipc export <handle-file>   allocate, fill, publish the handle, wait, verify
//   ipc import <handle-file>   open the handle, check what is there, write back
//
// Then both processes' kernels hammer one shared set of counters with atomics
// at the same time, as NVSHMEM's PEs add into one another's heaps, and no
// update may be lost: an atomic is atomic against another process's too.
//
// The two processes talk only through the handle file and the shared buffer:
// nothing about this works unless the memory really is shared.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)

static const int kInts = 1024;

// Adds `add` to every element, so each side can leave a mark the other sees.
__global__ void bump(int* p, int n, int add) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += add;
}

// Atomics on the shared counters: kRaceBlocks x kRaceThreads threads, kRaceIters each.
static const int kRaceBlocks = 32, kRaceThreads = 128, kRaceIters = 16;
static const unsigned long long kRacePerProcess = 1ull * kRaceBlocks * kRaceThreads * kRaceIters;
struct Counters {
  unsigned u32;            // atomicAdd
  unsigned cas;            // an atomicCAS loop
  unsigned long long u64;  // atomicAdd, 64-bit
  float f32;               // atomicAdd on whole numbers, exact below 2^24
};
__global__ void hammer(Counters* c) {
  for (int i = 0; i < kRaceIters; ++i) {
    atomicAdd(&c->u32, 1u);
    atomicAdd(&c->u64, 1ull);
    atomicAdd(&c->f32, 1.0f);
    unsigned old = *(volatile unsigned*)&c->cas;
    for (;;) {
      const unsigned got = atomicCAS(&c->cas, old, old + 1);
      if (got == old) break;
      old = got;
    }
  }
}

static bool write_file(const std::string& path, const void* data, size_t n) {
  const std::string tmp = path + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  const bool ok = fwrite(data, 1, n, f) == n;
  fclose(f);
  return ok && rename(tmp.c_str(), path.c_str()) == 0;
}

static bool read_file(const std::string& path, void* data, size_t n) {
  for (int i = 0; i < 600; ++i) {   // the other process may not have written it yet
    FILE* f = fopen(path.c_str(), "rb");
    if (f) {
      const bool ok = fread(data, 1, n, f) == n;
      fclose(f);
      if (ok) return true;
    }
    usleep(100000);
  }
  return false;
}

static void touch(const std::string& path) { write_file(path, "x", 1); }
static bool wait_for(const std::string& path) {
  for (int i = 0; i < 600; ++i) {
    if (access(path.c_str(), F_OK) == 0) return true;
    usleep(100000);
  }
  return false;
}

static int do_export(const std::string& handle_file) {
  int* buf = nullptr;
  CK(cudaMalloc(reinterpret_cast<void**>(&buf), kInts * sizeof(int)));
  Counters* counters = nullptr;
  CK(cudaMalloc(reinterpret_cast<void**>(&counters), sizeof(Counters)));
  CK(cudaMemset(counters, 0, sizeof(Counters)));
  cudaIpcMemHandle_t counters_handle{};
  CK(cudaIpcGetMemHandle(&counters_handle, counters));
  if (!write_file(handle_file + ".counters", &counters_handle, sizeof counters_handle)) {
    printf("FAIL could not publish the counters' handle\n");
    return 1;
  }
  int host[kInts];
  for (int i = 0; i < kInts; ++i) host[i] = i;
  CK(cudaMemcpy(buf, host, sizeof host, cudaMemcpyHostToDevice));

  cudaIpcMemHandle_t handle{};
  CK(cudaIpcGetMemHandle(&handle, buf));

  // The exporting process keeps using its own pointer: a kernel of its own runs
  // over the buffer after it was exported.
  bump<<<(kInts + 255) / 256, 256>>>(buf, kInts, 1000);
  CK(cudaDeviceSynchronize());

  // A handle for something that is not the base of an allocation is refused.
  // cudaErrorInvalidValue is what the call documents for a bad pointer.
  cudaIpcMemHandle_t bad{};
  WANT(cudaIpcGetMemHandle(&bad, buf + 4), cudaErrorInvalidValue);
  // And this process cannot import its own export.
  void* mine = nullptr;
  WANT(cudaIpcOpenMemHandle(&mine, handle, cudaIpcMemLazyEnablePeerAccess),
       cudaErrorInvalidValue);

  // An event recorded here, published for the other process to wait on.
  cudaEvent_t event = nullptr;
  CK(cudaEventCreate(&event));
  CK(cudaEventRecord(event, 0));
  cudaIpcEventHandle_t event_handle{};
  CK(cudaIpcGetEventHandle(&event_handle, event));
  if (!write_file(handle_file + ".event", &event_handle, sizeof event_handle)) {
    printf("FAIL could not publish the event handle\n");
    return 1;
  }
  if (!write_file(handle_file, &handle, sizeof handle)) {
    printf("FAIL could not publish the handle\n");
    return 1;
  }
  // The race: both processes start hammering once both are ready.
  touch(handle_file + ".xready");
  if (!wait_for(handle_file + ".iready")) {
    printf("FAIL the importing process never got ready to race\n");
    return 1;
  }
  hammer<<<kRaceBlocks, kRaceThreads>>>(counters);
  CK(cudaDeviceSynchronize());
  if (!wait_for(handle_file + ".done")) {
    printf("FAIL the importing process never finished\n");
    return 1;
  }
  Counters got{};
  CK(cudaMemcpy(&got, counters, sizeof got, cudaMemcpyDeviceToHost));
  const unsigned long long want = 2 * kRacePerProcess;
  if (got.u32 != want || got.cas != want || got.u64 != want ||
      static_cast<unsigned long long>(got.f32) != want) {
    printf("FAIL lost atomic updates between processes: u32 %u, CAS loop %u, u64 %llu, f32 %.0f; "
           "expected %llu each\n", got.u32, got.cas, got.u64, got.f32, want);
    return 1;
  }
  CK(cudaFree(counters));

  // What the other process wrote is here, in this process's own pointer.
  CK(cudaMemcpy(host, buf, sizeof host, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kInts; ++i) {
    const int want = i + 1000 + 7;   // this process added 1000, the other 7
    if (host[i] != want) {
      printf("FAIL exporter sees host[%d] = %d, expected %d\n", i, host[i], want);
      return 1;
    }
  }
  CK(cudaEventDestroy(event));
  CK(cudaFree(buf));
  printf("PASS export\n");
  return 0;
}

static int do_import(const std::string& handle_file) {
  // IPC_IMPORT_DEVICE: import with another device current than the exporter
  // used -- one GPU per process, as NCCL and NVSHMEM run -- so that this
  // process's kernels reach the memory on a device that did not export it.
  if (const char* d = getenv("IPC_IMPORT_DEVICE")) CK(cudaSetDevice(atoi(d)));
  cudaIpcMemHandle_t handle{};
  if (!read_file(handle_file, &handle, sizeof handle)) {
    printf("FAIL no handle to import\n");
    return 1;
  }
  void* ptr = nullptr;
  // The flag is not optional: it is the only one the API takes.
  WANT(cudaIpcOpenMemHandle(&ptr, handle, 0), cudaErrorInvalidValue);
  CK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));

  int* shared = static_cast<int*>(ptr);
  int host[kInts];
  CK(cudaMemcpy(host, shared, sizeof host, cudaMemcpyDeviceToHost));
  for (int i = 0; i < kInts; ++i) {
    if (host[i] != i + 1000) {   // what the exporter put there, its kernel included
      printf("FAIL importer sees host[%d] = %d, expected %d\n", i, host[i], i + 1000);
      return 1;
    }
  }
  // A kernel in this process writes the shared buffer.
  bump<<<(kInts + 255) / 256, 256>>>(shared, kInts, 7);
  CK(cudaDeviceSynchronize());

  // The event the other process recorded. Every operation here finishes before
  // the call that started it returns, so by the time its handle could be read
  // the work is done -- and waiting on it succeeds rather than being skipped.
  cudaIpcEventHandle_t event_handle{};
  if (!read_file(handle_file + ".event", &event_handle, sizeof event_handle)) {
    printf("FAIL no event handle to import\n");
    return 1;
  }
  cudaEvent_t theirs = nullptr;
  CK(cudaIpcOpenEventHandle(&theirs, event_handle));
  CK(cudaEventQuery(theirs));                 // complete
  CK(cudaEventSynchronize(theirs));
  CK(cudaStreamWaitEvent(0, theirs, 0));
  CK(cudaEventDestroy(theirs));
  // A handle this process exported cannot be opened here either.
  cudaEvent_t own = nullptr;
  cudaIpcEventHandle_t own_handle{};
  CK(cudaEventCreate(&own));
  CK(cudaIpcGetEventHandle(&own_handle, own));
  WANT(cudaIpcOpenEventHandle(&theirs, own_handle), cudaErrorInvalidValue);
  CK(cudaEventDestroy(own));

  CK(cudaIpcCloseMemHandle(ptr));
  WANT(cudaIpcCloseMemHandle(ptr), cudaErrorInvalidValue);   // closed once only

  // The race on the exporter's counters, alongside its own kernel.
  cudaIpcMemHandle_t counters_handle{};
  if (!read_file(handle_file + ".counters", &counters_handle, sizeof counters_handle)) {
    printf("FAIL no counters' handle to import\n");
    return 1;
  }
  void* counters = nullptr;
  CK(cudaIpcOpenMemHandle(&counters, counters_handle, cudaIpcMemLazyEnablePeerAccess));
  touch(handle_file + ".iready");
  if (!wait_for(handle_file + ".xready")) {
    printf("FAIL the exporting process never got ready to race\n");
    return 1;
  }
  hammer<<<kRaceBlocks, kRaceThreads>>>(static_cast<Counters*>(counters));
  CK(cudaDeviceSynchronize());
  CK(cudaIpcCloseMemHandle(counters));
  touch(handle_file + ".done");
  printf("PASS import\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("FAIL usage: ipc export|import <handle-file>\n");
    return 2;
  }
  const std::string mode = argv[1], handle_file = argv[2];
  if (mode == "export") return do_export(handle_file);
  if (mode == "import") return do_import(handle_file);
  printf("FAIL unknown mode %s\n", mode.c_str());
  return 2;
}
