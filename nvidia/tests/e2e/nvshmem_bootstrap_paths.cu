// NVSHMEM's bootstraps through a launcher, as an MPI job of two PEs: a communicator
// (NVSHMEMX_INIT_WITH_MPI_COMM), the variables NVSHMEM_BOOTSTRAP=MPI, plugin, PMI with
// NVSHMEM_BOOTSTRAP_PMI=PMIX, PMI and PMI-2 (which a machine without libpmi.so and
// libpmi2.so answers with a job of one PE), and the settings NVSHMEM refuses.
//
// Built with NVIDIA's NVSHMEM headers and device library, like nvshmem_device.cu:
//   mpirun -np 2 ./nvshmem_bootstrap_paths <scenario>
// Both PEs use device 0 (NVIDIA's NVSHMEM does not start PEs on two GeForce GPUs without
// peer access, and on one GPU it is in its multiple-processes-per-GPU mode: status 3), except in
// the scenario two_gpus, where PE r uses device r: that is a job NVSHMEM starts only on GPUs that can
// reach each other (an RTX 3060 pair cannot: "Peer GPU 1 is not accessible", exit status 255).
// Each PE writes what it saw to $NVSHMEM_BOOT_OUT.<rank>; run_nvshmem_bootstrap.sh
// collects them and compares them with what NVIDIA's NVSHMEM 3.8 wrote on an RTX 3060.
#include <mpi.h>

#include <cuda_runtime_api.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
  const std::string scenario = argc > 1 ? argv[1] : "attr";
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(scenario == "two_gpus" ? rank : 0);
  std::string out;
  char line[256];
  auto say = [&](const char* fmt, auto... args) {
    std::snprintf(line, sizeof line, fmt, args...);
    out += line;
    out += '\n';
  };
  say("before init: status %d", nvshmemx_init_status());
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm comm = MPI_COMM_NULL;
  if (scenario == "attr" || scenario == "two_gpus") {
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    say("set_attr_mpi_comm_args: %d", nvshmemx_set_attr_mpi_comm_args(&comm, &attr));
    say("init_attr(MPI_COMM): %d", nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
  } else if (scenario == "attr_reversed") {  // the PE numbers are the communicator's ranks
    MPI_Comm_split(MPI_COMM_WORLD, 0, size - rank, &comm);
    say("set_attr_mpi_comm_args: %d", nvshmemx_set_attr_mpi_comm_args(&comm, &attr));
    say("init_attr(MPI_COMM): %d", nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
  } else if (scenario == "attr_null") {  // no communicator in the attributes: MPI_COMM_WORLD
    say("init_attr(MPI_COMM): %d", nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
  } else {  // the variables NVSHMEM_BOOTSTRAP, NVSHMEM_BOOTSTRAP_PMI and NVSHMEM_BOOTSTRAP_PLUGIN decide
    nvshmem_init();
  }
  // From here the job is up; a failed init has already ended the program with status 255.
  say("after init: status %d", nvshmemx_init_status());
  const int me = nvshmem_my_pe(), n = nvshmem_n_pes();
  say("PE %d of %d, %d on this node", me, n, nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE));
  int* p = static_cast<int*>(nvshmem_malloc(256));
  say("nvshmem_malloc: %s", p ? "memory" : "NULL");
  if (p) {  // each PE's word holds 100 + its PE; read the next PE's
    const int mine = 100 + me;
    cudaMemcpy(p, &mine, sizeof mine, cudaMemcpyHostToDevice);
    nvshmem_barrier_all();
    say("read from the next PE: %d", nvshmem_int_g(p, (me + 1) % n));
    nvshmem_barrier_all();
    nvshmem_free(p);
  }
  nvshmem_finalize();
  // The bootstraps of a launcher stay bootstrapped (1); a unique ID's does not (0).
  say("after finalize: status %d", nvshmemx_init_status());
  const char* prefix = std::getenv("NVSHMEM_BOOT_OUT");
  if (prefix) {
    FILE* f = std::fopen((std::string(prefix) + "." + std::to_string(rank)).c_str(), "w");
    if (f) {
      std::fputs(out.c_str(), f);
      std::fclose(f);
    }
  }
  if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
  MPI_Finalize();
  return 0;
}
