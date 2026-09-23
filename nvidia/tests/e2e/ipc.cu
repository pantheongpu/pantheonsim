// Device memory shared between two processes, which is what CUDA IPC is for:
// one process allocates and fills a buffer, hands its handle to another, and
// the second one's kernel reads and writes the same bytes. vLLM shares a KV
// cache this way; NCCL uses it for peer buffers on one machine.
//
//   ipc export <handle-file>   allocate, fill, publish the handle, wait, verify
//   ipc import <handle-file>   open the handle, check what is there, write back
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
  if (!wait_for(handle_file + ".done")) {
    printf("FAIL the importing process never finished\n");
    return 1;
  }

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
