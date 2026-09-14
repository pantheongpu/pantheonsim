// Multi-GPU semantics, differentially: device enumeration, per-device context
// and allocation, kernels on each device, peer-to-peer copies, and stream and
// event objects that belong to one device.
//
// Everything printed here must hold on any machine with at least two GPUs, so
// the same output is expected from a real multi-GPU box and from a VirtualGPU
// rack of the same size. Topology-dependent facts (whether NVLink exists,
// whether cudaDeviceCanAccessPeer is set) are deliberately reported as
// behaviour -- "the copy produced the right bytes" -- rather than as raw flags,
// because those flags differ between two machines with the same GPU in them.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("%s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

__global__ void ramp(float* p, int n, float base, float step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = base + step * i;
}

__global__ void scale_in_place(float* p, int n, float k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= k;
}

// Reduce on the device so the check exercises the device that owns the memory,
// not just a copy back to the host.
__global__ void checksum(const float* p, int n, double* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += p[i];
    *out = s;
  }
}

static double read_checksum(const float* d, int n) {
  double* dsum = nullptr;
  cudaMalloc(&dsum, sizeof(double));
  checksum<<<1, 1>>>(d, n, dsum);
  cudaDeviceSynchronize();
  double h = 0;
  cudaMemcpy(&h, dsum, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dsum);
  return h;
}

int main() {
  int ndev = 0;
  CU(cudaGetDeviceCount(&ndev));
  if (ndev < 2) { printf("SKIP: %d device(s), need 2\n", ndev); return 0; }
  const int use = 2;   // two is enough to prove the semantics; more is the same
  printf("devices at least 2: yes\n");

  // A homogeneous machine must report the same device for every ordinal; the
  // simulator builds its rack from one profile, so this has to hold there too.
  cudaDeviceProp p0{}, p1{};
  CU(cudaGetDeviceProperties(&p0, 0));
  CU(cudaGetDeviceProperties(&p1, 1));
  printf("homogeneous: %s, cc equal: %s, mem equal: %s\n",
         std::string(p0.name) == p1.name ? "yes" : "no",
         (p0.major == p1.major && p0.minor == p1.minor) ? "yes" : "no",
         p0.totalGlobalMem == p1.totalGlobalMem ? "yes" : "no");
  printf("distinct pci slots: %s\n",
         (p0.pciBusID != p1.pciBusID || p0.pciDeviceID != p1.pciDeviceID) ? "yes" : "no");

  const int n = 4096;
  std::vector<float*> buf(use, nullptr);
  std::vector<cudaStream_t> streams(use);

  // A separate allocation and stream per device, filled by a kernel that runs
  // on that device.
  for (int d = 0; d < use; ++d) {
    CU(cudaSetDevice(d));
    CU(cudaStreamCreate(&streams[d]));
    CU(cudaMalloc(&buf[d], n * sizeof(float)));
    ramp<<<(n + 255) / 256, 256, 0, streams[d]>>>(buf[d], n, 1.0f + d, 0.5f + d);
    CU(cudaStreamSynchronize(streams[d]));
  }
  for (int d = 0; d < use; ++d) {
    CU(cudaSetDevice(d));
    printf("device %d checksum %.1f\n", d, read_checksum(buf[d], n));
  }

  // cudaSetDevice must be a per-thread binding that survives the calls above.
  int cur = -1;
  CU(cudaSetDevice(1));
  CU(cudaGetDevice(&cur));
  printf("current device sticks: %s\n", cur == 1 ? "yes" : "no");

  // Allocations belong to one device: touching device 0's buffer must leave
  // device 1's alone.
  CU(cudaSetDevice(0));
  scale_in_place<<<(n + 255) / 256, 256>>>(buf[0], n, 0.0f);
  CU(cudaDeviceSynchronize());
  CU(cudaSetDevice(0));
  const double after0 = read_checksum(buf[0], n);
  CU(cudaSetDevice(1));
  const double after1 = read_checksum(buf[1], n);
  printf("after zeroing d0: d0=%.1f d1=%.1f\n", after0, after1);

  // Peer copy. cudaMemcpyPeer is defined whether or not peer access is
  // enabled -- without it the driver routes through the host -- so the bytes
  // must arrive either way, and that is what gets checked rather than the
  // capability flag.
  int can01 = 0, can10 = 0;
  CU(cudaDeviceCanAccessPeer(&can01, 0, 1));
  CU(cudaDeviceCanAccessPeer(&can10, 1, 0));
  printf("peer capability symmetric: %s\n", can01 == can10 ? "yes" : "no");
  if (can01) {
    CU(cudaSetDevice(0));
    cudaError_t e = cudaDeviceEnablePeerAccess(1, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
      printf("cudaDeviceEnablePeerAccess -> %s\n", cudaGetErrorString(e));
      return 1;
    }
  }
  float* dst = nullptr;
  CU(cudaSetDevice(0));
  CU(cudaMalloc(&dst, n * sizeof(float)));
  CU(cudaMemcpyPeer(dst, 0, buf[1], 1, n * sizeof(float)));
  CU(cudaDeviceSynchronize());
  printf("peer copy d1->d0 checksum %.1f\n", read_checksum(dst, n));

  // The same copy on a stream, and an event recorded on the destination device.
  CU(cudaMemsetAsync(dst, 0, n * sizeof(float), streams[0]));
  CU(cudaMemcpyPeerAsync(dst, 0, buf[1], 1, n * sizeof(float), streams[0]));
  cudaEvent_t done;
  CU(cudaEventCreate(&done));
  CU(cudaEventRecord(done, streams[0]));
  CU(cudaEventSynchronize(done));
  printf("async peer copy checksum %.1f\n", read_checksum(dst, n));
  CU(cudaEventDestroy(done));

  // Device-to-device through the generic entry point, which has to work too.
  CU(cudaMemset(dst, 0, n * sizeof(float)));
  CU(cudaMemcpy(dst, buf[1], n * sizeof(float), cudaMemcpyDeviceToDevice));
  printf("cudaMemcpyDeviceToDevice checksum %.1f\n", read_checksum(dst, n));

  // A kernel launched on device 0 must not be able to touch device 1 memory
  // when peer access is off; when it is on, the read is legal. Either way the
  // launch itself must not silently corrupt anything -- verify device 1 again.
  CU(cudaSetDevice(1));
  printf("device 1 intact after peer traffic: %.1f\n", read_checksum(buf[1], n));

  for (int d = 0; d < use; ++d) {
    CU(cudaSetDevice(d));
    CU(cudaStreamDestroy(streams[d]));
    CU(cudaFree(buf[d]));
  }
  CU(cudaSetDevice(0));
  CU(cudaFree(dst));
  printf("RESULT: multi-device semantics consistent\n");
  return 0;
}
