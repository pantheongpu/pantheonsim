// cuFile's calls new in CUDA 13.2's library (cuFile 1.16 and 1.17): the GPU
// bounce-buffer slab array, the P2P flags and the PCIe topology export. The
// sequence below is the order the calls are made in, because the library's
// answers depend on what has loaded its configuration: and every status is
// what CUDA 13.2's libcufile.so.0 answered on an RTX 3060 in compatibility
// mode (no nvidia-fs). The program passes against it.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "../../include/vgpu_cufile.h"

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call).err == (int)(want), #call " -> " #want)

int main() {
  size_t sz[8] = {}, ct[8] = {};
  CUfileP2PFlags_t f = (CUfileP2PFlags_t)-7;

  // Nothing has loaded the configuration and nothing is set.
  IS(cuFileGetParameterGpuBounceBufferSlabArray(sz, ct, 4), CU_FILE_INVALID_VALUE);
  IS(cuFileSetParameterGpuBounceBufferSlabArray(nullptr, ct, 3), CU_FILE_INVALID_VALUE);
  IS(cuFileSetParameterGpuBounceBufferSlabArray(sz, nullptr, 3), CU_FILE_INVALID_VALUE);
  IS(cuFileSetParameterGpuBounceBufferSlabArray(sz, ct, 0), CU_FILE_INVALID_VALUE);
  IS(cuFileSetParameterGpuBounceBufferSlabArray(sz, ct, -1), CU_FILE_INVALID_VALUE);

  // A set is kept as given and read back at the length it was set with.
  const size_t gs[3] = {4, 16, 64}, gc[3] = {5, 6, 7};
  IS(cuFileSetParameterGpuBounceBufferSlabArray(gs, gc, 3), CU_FILE_SUCCESS);
  IS(cuFileGetParameterGpuBounceBufferSlabArray(sz, ct, 3), CU_FILE_SUCCESS);
  check(sz[0] == 4 && sz[1] == 16 && sz[2] == 64 && ct[0] == 5 && ct[1] == 6 && ct[2] == 7, "the slab array reads back");
  IS(cuFileGetParameterGpuBounceBufferSlabArray(sz, ct, 4), CU_FILE_INVALID_VALUE);
  IS(cuFileGetParameterGpuBounceBufferSlabArray(sz, ct, 2), CU_FILE_INVALID_VALUE);
  IS(cuFileGetParameterGpuBounceBufferSlabArray(nullptr, ct, 3), CU_FILE_INVALID_VALUE);
  IS(cuFileGetParameterGpuBounceBufferSlabArray(sz, ct, 0), CU_FILE_INVALID_VALUE);
  // No check on what is set: an unaligned, descending array with a zero count.
  const size_t odd[3] = {16, 5, 64}, zeros[3] = {1, 0, 3};
  IS(cuFileSetParameterGpuBounceBufferSlabArray(odd, zeros, 3), CU_FILE_SUCCESS);
  IS(cuFileGetParameterGpuBounceBufferSlabArray(sz, ct, 3), CU_FILE_SUCCESS);
  check(sz[0] == 16 && sz[1] == 5 && ct[1] == 0, "an unchecked array reads back as given");
  IS(cuFileSetParameterGpuBounceBufferSlabArray(gs, gc, 3), CU_FILE_SUCCESS);

  // P2P flags: nothing set at first, one flag takes, the rest are refused.
  IS(cuFileDriverGetP2PFlags(CU_FILE_NVME_SUPPORTED, &f), CU_FILE_SUCCESS);
  check((int)f == 0, "no P2P flag is set at first");
  IS(cuFileDriverGetP2PFlags(CU_FILE_NVME_SUPPORTED, nullptr), CU_FILE_INVALID_VALUE);
  IS(cuFileDriverGetP2PFlags(CU_FILE_MAX_TARGET_TYPES, &f), CU_FILE_INVALID_VALUE);
  IS(cuFileDriverSetP2PFlags(CU_FILE_MAX_TARGET_TYPES, CU_FILE_P2P_FLAG_PCI_P2PDMA), CU_FILE_INVALID_VALUE);
  IS(cuFileDriverSetP2PFlags(CU_FILE_NVME_SUPPORTED, CU_FILE_P2P_FLAG_PCI_P2PDMA), CU_FILE_SUCCESS);
  IS(cuFileDriverGetP2PFlags(CU_FILE_NVME_SUPPORTED, &f), CU_FILE_SUCCESS);
  check((int)f == 1, "the P2PDMA flag is set");
  IS(cuFileDriverGetP2PFlags(CU_FILE_LUSTRE_SUPPORTED, &f), CU_FILE_SUCCESS);
  check((int)f == 0, "another file system's flags are its own");
  for (int bad : {2, 4, 8, 3, 15, 16}) {
    char what[64];
    std::snprintf(what, sizeof what, "P2P flags 0x%x are refused", bad);
    check((int)cuFileDriverSetP2PFlags(CU_FILE_NVME_SUPPORTED, (CUfileP2PFlags_t)bad).err == CU_FILE_INVALID_VALUE, what);
  }
  IS(cuFileDriverSetP2PFlags(CU_FILE_NVME_SUPPORTED, (CUfileP2PFlags_t)0), CU_FILE_SUCCESS);
  IS(cuFileDriverGetP2PFlags(CU_FILE_NVME_SUPPORTED, &f), CU_FILE_SUCCESS);
  check((int)f == 1, "setting no flags changes nothing");

  // The P2P calls loaded the configuration: it can no longer be set.
  IS(cuFileSetParameterGpuBounceBufferSlabArray(gs, gc, 3), CU_FILE_DRIVER_ALREADY_OPEN);
  IS(cuFileSetParameterSizeT(CUFILE_PARAM_EXECUTION_MAX_IO_THREADS, 8), CU_FILE_DRIVER_ALREADY_OPEN);
  check(cuFileUseCount() == 0, "...though the driver was not opened");

  // The topology export writes a file, leaves the configuration settable,
  // and forgets the P2P flags.
  const std::string path = "cufile_tail_topology.json";
  IS(cuFileExportPCIeTopology(nullptr), CU_FILE_INVALID_VALUE);
  IS(cuFileExportPCIeTopology(""), CU_FILE_INVALID_VALUE);
  IS(cuFileExportPCIeTopology("/nonexistent-directory/topology.json"), CU_FILE_INVALID_VALUE);
  IS(cuFileExportPCIeTopology("/"), CU_FILE_INVALID_VALUE);
  IS(cuFileExportPCIeTopology(path.c_str()), CU_FILE_SUCCESS);
  IS(cuFileExportPCIeTopology(path.c_str()), CU_FILE_SUCCESS);  // overwritten
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  check(text.rfind("// Topology file for cuFile.", 0) == 0 && text.size() > 4 && text.compare(text.size() - 4, 4, "null") == 0,
        "the exported file is the topology header and null");
  std::remove(path.c_str());
  IS(cuFileDriverGetP2PFlags(CU_FILE_NVME_SUPPORTED, &f), CU_FILE_SUCCESS);
  // (this call loads the configuration again)
  check((int)f == 0, "the export forgot the P2P flags");
  IS(cuFileExportPCIeTopology(nullptr), CU_FILE_INVALID_VALUE);
  const size_t g8[8] = {4, 8, 12, 16, 20, 24, 28, 32}, c8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  IS(cuFileExportPCIeTopology("/dev/null"), CU_FILE_SUCCESS);
  IS(cuFileSetParameterGpuBounceBufferSlabArray(g8, c8, 8), CU_FILE_SUCCESS);

  // An open driver: its configuration is fixed, P2P flags work, and a final
  // close forgets them but keeps the slab array.
  IS(cuFileDriverOpen(), CU_FILE_SUCCESS);
  IS(cuFileSetParameterGpuBounceBufferSlabArray(gs, gc, 3), CU_FILE_DRIVER_ALREADY_OPEN);
  IS(cuFileGetParameterGpuBounceBufferSlabArray(sz, ct, 8), CU_FILE_SUCCESS);
  check(sz[0] == 4 && sz[7] == 32 && ct[7] == 1, "the open driver uses the array that was set");
  IS(cuFileGetParameterGpuBounceBufferSlabArray(sz, ct, 4), CU_FILE_INVALID_VALUE);
  IS(cuFileDriverSetP2PFlags(CU_FILE_LUSTRE_SUPPORTED, CU_FILE_P2P_FLAG_PCI_P2PDMA), CU_FILE_SUCCESS);
  IS(cuFileDriverGetP2PFlags(CU_FILE_LUSTRE_SUPPORTED, &f), CU_FILE_SUCCESS);
  check((int)f == 1, "a flag is set on an open driver");
  IS(cuFileDriverClose(), CU_FILE_SUCCESS);
  IS(cuFileDriverGetP2PFlags(CU_FILE_LUSTRE_SUPPORTED, &f), CU_FILE_SUCCESS);
  check((int)f == 0, "a final close forgot the flag");
  IS(cuFileGetParameterGpuBounceBufferSlabArray(sz, ct, 8), CU_FILE_SUCCESS);
  check(sz[0] == 4 && ct[0] == 1, "...and kept the slab array");

  std::printf(failures ? "FAIL: %d cuFile checks\n" : "PASS: every cuFile 1.16-1.17 check\n", failures);
  return failures ? 1 : 0;
}
