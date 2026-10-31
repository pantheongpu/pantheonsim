# AMD sources

The HIP runtime (`hip_api.cpp`), the HSA runtime (`hsa_api.cpp`), the GCN, CDNA
and RDNA decoder and executor (`gcn_decode.cpp`, `gcn_exec.cpp`), the code object
and bundle readers, ROCm SMI and AMD SMI, and the rocprofiler-sdk shim
(`rocprofiler_sdk.cpp`) are built from here, as `nvidia/src/` holds the CUDA
driver, runtime and libraries. [amd/README.md](../README.md) says what each covers.
