"""Every function the toolchain's nvml.h declares is exported by libnvidia-ml.so.1.

usage: nvml_coverage.py <shim dir> <nvml.h>

The names come from the header the shim was built against, so the check follows
the toolchain: CUDA 12.0's header declares 307 functions and this checks those,
CUDA 13.2's declares 408. A name the shim lacks is a symbol pynvml (and every tool
that looks entry points up by name) turns into FunctionNotFound.
"""
import ctypes, os, re, sys

shim, header = sys.argv[1], sys.argv[2]
text = open(header, errors="ignore").read()
text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
text = re.sub(r"//[^\n]*", "", text)
names = sorted(set(m.group(1) for m in re.finditer(
    r"(?:nvmlReturn_t|const char\s*\*|unsigned int)\s+DECLDIR\s+(nvml\w+)\s*\(", text)))
lib = ctypes.CDLL(os.path.join(shim, "libnvidia-ml.so.1"))
missing = [n for n in names if not hasattr(lib, n)]
print(f"{len(names)} functions declared in {header}, {len(names) - len(missing)} exported")
for n in missing:
    print("MISSING", n)
# The unversioned and versioned spellings NVIDIA exports side by side: both are
# looked up, by pynvml and by the CUDA toolkit's own tools.
spellings = ["nvmlInit", "nvmlInit_v2", "nvmlDeviceGetCount", "nvmlDeviceGetCount_v2",
             "nvmlDeviceGetHandleByIndex", "nvmlDeviceGetHandleByIndex_v2",
             "nvmlDeviceGetHandleByPciBusId", "nvmlDeviceGetHandleByPciBusId_v2",
             "nvmlDeviceGetPciInfo", "nvmlDeviceGetPciInfo_v2", "nvmlDeviceGetPciInfo_v3",
             "nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetComputeRunningProcesses_v2",
             "nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetNvLinkRemotePciInfo",
             "nvmlDeviceGetNvLinkRemotePciInfo_v2", "nvmlDeviceGetGridLicensableFeatures",
             "nvmlDeviceGetGridLicensableFeatures_v4", "nvmlComputeInstanceGetInfo",
             "nvmlComputeInstanceGetInfo_v2", "nvmlDeviceGetGpuInstancePossiblePlacements",
             "nvmlDeviceGetGpuInstancePossiblePlacements_v2", "nvmlDeviceRemoveGpu", "nvmlDeviceRemoveGpu_v2",
             "nvmlDeviceGetAttributes", "nvmlDeviceGetAttributes_v2"]
lost = [n for n in spellings if n in names and not hasattr(lib, n)]
if lost:
    print("MISSING spellings:", lost)
    missing += lost
print("FAIL" if missing else "ok")
sys.exit(1 if missing else 0)
