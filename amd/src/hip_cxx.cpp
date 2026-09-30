// The C++ functions ROCm's libamdhip64 exports beside its C API: hip_ext.h
// declares hipExtModuleLaunchKernel and hipHccModuleLaunchKernel for C++ as
// well, so a program compiled as C++ can bind to their mangled names. Each is
// the C function of the same name (hip_api.cpp), reached here through its
// symbol, since one translation unit cannot hold both.
#include <cstddef>
#include <cstdint>

struct ihipModuleSymbol_t;
struct ihipStream_t;
struct ihipEvent_t;

using Error = int;   // hipError_t
Error vgpu_ext_launch(ihipModuleSymbol_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t,
                 ihipStream_t*, void**, void**, ihipEvent_t*, ihipEvent_t*, uint32_t) __asm__("hipExtModuleLaunchKernel");
Error vgpu_hcc_launch(ihipModuleSymbol_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t,
                 ihipStream_t*, void**, void**, ihipEvent_t*, ihipEvent_t*) __asm__("hipHccModuleLaunchKernel");

Error hipExtModuleLaunchKernel(ihipModuleSymbol_t* f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx,
                               uint32_t ly, uint32_t lz, size_t shared, ihipStream_t* stream, void** params,
                               void** extra, ihipEvent_t* start, ihipEvent_t* stop, uint32_t flags) {
  return vgpu_ext_launch(f, gx, gy, gz, lx, ly, lz, shared, stream, params, extra, start, stop, flags);
}
Error hipHccModuleLaunchKernel(ihipModuleSymbol_t* f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx,
                               uint32_t ly, uint32_t lz, size_t shared, ihipStream_t* stream, void** params,
                               void** extra, ihipEvent_t* start, ihipEvent_t* stop) {
  return vgpu_hcc_launch(f, gx, gy, gz, lx, ly, lz, shared, stream, params, extra, start, stop);
}
