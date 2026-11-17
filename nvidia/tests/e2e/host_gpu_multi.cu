// The program run_host_gpu_multi_stub.sh runs (docs/perf-design/12-host-gpu-turbo.md section 17): kernels on several simulated devices, for turbo mode on a machine
// with several cards (the stand-in driver reports as many as the script says). `bump` adds 1 to four ints; the stand-in card does the same to the first four ints
// of a launch's first pointer, so a launch that went to a card and one the simulator ran leave the same bytes, and any mix-up of memory between devices, cards or
// the simulator shows in what the program prints (which must not depend on the mode).
//   dp        every device in turn: 4 ints set from the host, 3 launches of `bump`, read back and printed.
//   copy      device 0: set, one bump; cudaMemcpyPeer to device 1 (no peer access enabled: the runtime copies through memory the simulator holds); two bumps on
//             device 1; read back. The card's bump on device 0 must reach the copy: the simulator pulls it from card 0 first.
//   peerptr   device 1 launches `bump` on device 0's pointer with no peer access: the simulator's invalid-pointer error, and the launch never reaches a card
//             (the card has only its own window; an access outside it is a fault, which kills every context of the process).
//   peeron    peer access enabled between devices 0 and 1 (the simulated machine's profile allows it), then the bumps of `dp` on both: launches on devices
//             linked by peer access stay in the simulator (MemoryManager::card_blocker).
//   solo      one device (run with one simulated device): three launches and a read-back; the launches are waited for, as without several devices.
//   mixed     `dp`, then a program-level cudaMemcpyPeer from the LAST device to device 0 and one more bump on device 0: the last device may be the one with no card.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void bump(int* p, int n) {
  const int i = threadIdx.x;
  if (i < n) p[i] += 1;
}

static int g_n = 0;
static void show(const char* tag, int d, const int* v) { std::printf("%s dev %d: %d %d %d %d\n", tag, d, v[0], v[1], v[2], v[3]); }
static int* make(int d, int base) {
  int* p = nullptr;
  cudaSetDevice(d);
  cudaMalloc(&p, 64 * sizeof(int));
  int h[4] = {base, base + 10, base + 20, base + 30};
  cudaMemcpy(p, h, sizeof h, cudaMemcpyHostToDevice);
  return p;
}
static void read_back(const char* tag, int d, int* p) {
  int h[4] = {};
  cudaSetDevice(d);
  cudaDeviceSynchronize();
  cudaMemcpy(h, p, sizeof h, cudaMemcpyDeviceToHost);
  show(tag, d, h);
}

int main(int argc, char** argv) {
  int* buf[8] = {};
  const char* mode = argc > 1 ? argv[1] : "dp";
  cudaGetDeviceCount(&g_n);
  std::printf("devices %d\n", g_n);
  if (!std::strcmp(mode, "solo")) {   // one device, the usual program: three launches, read back
    buf[0] = make(0, 5);
    for (int i = 0; i < 3; ++i) bump<<<1, 32>>>(buf[0], 4);
    read_back("solo", 0, buf[0]);
    return 0;
  }
  if (g_n < 2) return 0;
  if (!std::strcmp(mode, "dp") || !std::strcmp(mode, "mixed") || !std::strcmp(mode, "peeron")) {
    if (!std::strcmp(mode, "peeron")) {
      cudaSetDevice(0);
      std::printf("enable 0->1: %s\n", cudaGetErrorName(cudaDeviceEnablePeerAccess(1, 0)));
      cudaSetDevice(1);
      std::printf("enable 1->0: %s\n", cudaGetErrorName(cudaDeviceEnablePeerAccess(0, 0)));
    }
    const int n = g_n < 8 ? g_n : 8;
    for (int d = 0; d < n; ++d) buf[d] = make(d, d * 100);
    for (int round = 0; round < 3; ++round)
      for (int d = 0; d < n; ++d) {
        cudaSetDevice(d);
        bump<<<1, 32>>>(buf[d], 4);
      }
    for (int d = 0; d < n; ++d) read_back("dp", d, buf[d]);
    if (!std::strcmp(mode, "mixed")) {
      cudaMemcpyPeer(buf[0], 0, buf[n - 1], n - 1, 4 * sizeof(int));
      cudaSetDevice(0);
      bump<<<1, 32>>>(buf[0], 4);
      read_back("mixed", 0, buf[0]);
    }
  } else if (!std::strcmp(mode, "copy")) {
    buf[0] = make(0, 10);
    buf[1] = make(1, 500);
    cudaSetDevice(0);
    bump<<<1, 32>>>(buf[0], 4);
    cudaDeviceSynchronize();
    std::printf("peer copy: %s\n", cudaGetErrorName(cudaMemcpyPeer(buf[1], 1, buf[0], 0, 4 * sizeof(int))));
    cudaSetDevice(1);
    bump<<<1, 32>>>(buf[1], 4);
    bump<<<1, 32>>>(buf[1], 4);
    read_back("copy", 1, buf[1]);
    read_back("copy", 0, buf[0]);
  } else if (!std::strcmp(mode, "peerptr")) {
    buf[0] = make(0, 7);
    buf[1] = make(1, 9);   // (device 1 holds live memory of its own, so a launch on it has a window to be sent in)
    cudaSetDevice(1);
    bump<<<1, 32>>>(buf[0], 4);
    std::printf("peer pointer launch: %s / %s\n", cudaGetErrorName(cudaGetLastError()), cudaGetErrorName(cudaDeviceSynchronize()));
  }
  return 0;
}
