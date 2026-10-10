// NVSHMEM bootstrapped through OpenSHMEM (NVSHMEMX_INIT_WITH_SHMEM, or NVSHMEM_BOOTSTRAP=SHMEM):
//   oshrun -np 2 ./nvshmem_bootstrap_shmem <flag|env>
// What each PE saw goes to $NVSHMEM_BOOT_OUT.<rank>; see nvshmem_bootstrap_paths.cu.
#include <shmem.h>

#include <cuda_runtime_api.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "flag";
  shmem_init();
  const int rank = shmem_my_pe();
  cudaSetDevice(0);
  std::string out;
  char line[256];
  auto say = [&](const char* fmt, auto... args) {
    std::snprintf(line, sizeof line, fmt, args...);
    out += line;
    out += '\n';
  };
  say("before init: status %d", nvshmemx_init_status());
  if (mode == "flag") {
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    say("init_attr(SHMEM): %d", nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr));
  } else {
    nvshmem_init();
  }
  say("after init: status %d", nvshmemx_init_status());
  const int me = nvshmem_my_pe(), n = nvshmem_n_pes();
  say("PE %d of %d", me, n);
  int* p = static_cast<int*>(nvshmem_malloc(256));
  say("nvshmem_malloc: %s", p ? "memory" : "NULL");
  if (p) {
    const int mine = 100 + me;
    cudaMemcpy(p, &mine, sizeof mine, cudaMemcpyHostToDevice);
    nvshmem_barrier_all();
    say("read from the next PE: %d", nvshmem_int_g(p, (me + 1) % n));
    nvshmem_barrier_all();
    nvshmem_free(p);
  }
  nvshmem_finalize();
  say("after finalize: status %d", nvshmemx_init_status());
  if (const char* prefix = std::getenv("NVSHMEM_BOOT_OUT")) {
    if (FILE* f = std::fopen((std::string(prefix) + "." + std::to_string(rank)).c_str(), "w")) {
      std::fputs(out.c_str(), f);
      std::fclose(f);
    }
  }
  shmem_finalize();
  return 0;
}
