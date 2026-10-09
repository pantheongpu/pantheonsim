// Runtime API entry points the CUDA runtime exports and this one did not: a
// program that names one of them failed to link, or, loaded with -z now as
// PyTorch's libraries are, failed to start. Each is checked against what its
// documentation says it does.
//   cudaMemset3D(Async)                    cudaChooseDevice, cudaInitDevice
//   cudaDeviceGetByPCIBusId                cudaMemcpy2D{To,From}ArrayAsync, 2DArrayToArray
//   cudaGet{Texture,Surface}Object*Desc    cudaFuncGetName, cudaFuncGetParamInfo
//   cudaMemAdvise_v2, cudaMemPrefetchAsync_v2
//   cudaDeviceSet/GetSharedMemConfig, cudaThread* (deprecated spellings)
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "fatbin.hpp"
#include "vtest.hpp"

extern "C" {
void** __cudaRegisterFatBinary(void* fatCubin);
void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun,
                            const char* deviceName, int thread_limit, void* tid, void* bid, void* bDim,
                            void* gDim, int* wSize);
}

namespace {

// --- cudaMemset3D -----------------------------------------------------------

VTEST(memset3d_fills_the_box_and_leaves_the_padding) {
  cudaPitchedPtr p{};
  const cudaExtent alloc = make_cudaExtent(100, 4, 3);   // 100 bytes a row, 4 rows, 3 slices
  VCHECK_EQ(cudaMalloc3D(&p, alloc), cudaSuccess);
  VCHECK(p.pitch >= 100);
  const size_t total = p.pitch * 4 * 3;
  VCHECK_EQ(cudaMemset(p.ptr, 0x11, total), cudaSuccess);
  VCHECK_EQ(cudaMemset3D(p, 0xAB, make_cudaExtent(50, 3, 2)), cudaSuccess);
  std::vector<uint8_t> host(total);
  VCHECK_EQ(cudaMemcpy(host.data(), p.ptr, total, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t z = 0; z < 3; ++z)
    for (size_t y = 0; y < 4; ++y)
      for (size_t x = 0; x < p.pitch; ++x) {
        const bool inside = x < 50 && y < 3 && z < 2;
        VCHECK_EQ(host[(z * 4 + y) * p.pitch + x], inside ? 0xAB : 0x11);
      }
  VCHECK_EQ(cudaFree(p.ptr), cudaSuccess);
}

VTEST(memset3d_async_matches_the_synchronous_form) {
  cudaPitchedPtr p{};
  VCHECK_EQ(cudaMalloc3D(&p, make_cudaExtent(64, 8, 2)), cudaSuccess);
  VCHECK_EQ(cudaMemset(p.ptr, 0, p.pitch * 8 * 2), cudaSuccess);
  cudaStream_t stream;
  VCHECK_EQ(cudaStreamCreate(&stream), cudaSuccess);
  VCHECK_EQ(cudaMemset3DAsync(p, 0x5A, make_cudaExtent(64, 8, 2), stream), cudaSuccess);
  VCHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  std::vector<uint8_t> host(p.pitch * 8 * 2);
  VCHECK_EQ(cudaMemcpy(host.data(), p.ptr, host.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t z = 0; z < 2; ++z)
    for (size_t y = 0; y < 8; ++y)
      for (size_t x = 0; x < 64; ++x) VCHECK_EQ(host[(z * 8 + y) * p.pitch + x], 0x5A);
  VCHECK_EQ(cudaStreamDestroy(stream), cudaSuccess);
  VCHECK_EQ(cudaFree(p.ptr), cudaSuccess);
}

VTEST(memset3d_refuses_a_box_that_does_not_fit_and_accepts_an_empty_one) {
  cudaPitchedPtr p{};
  VCHECK_EQ(cudaMalloc3D(&p, make_cudaExtent(64, 4, 2)), cudaSuccess);
  VCHECK_EQ(cudaMemset3D(p, 0, make_cudaExtent(p.pitch + 1, 1, 1)), cudaErrorInvalidValue);   // wider than a row
  VCHECK_EQ(cudaMemset3D(p, 0, make_cudaExtent(8, 5, 2)), cudaErrorInvalidValue);              // taller than a slice
  cudaPitchedPtr none = p;
  none.ptr = nullptr;
  VCHECK_EQ(cudaMemset3D(none, 0, make_cudaExtent(8, 1, 1)), cudaErrorInvalidValue);
  VCHECK_EQ(cudaMemset3D(none, 0, make_cudaExtent(0, 1, 1)), cudaSuccess);   // nothing to do
  VCHECK_EQ(cudaFree(p.ptr), cudaSuccess);
}

// --- device selection -------------------------------------------------------

VTEST(choose_device_picks_the_device_matching_the_request) {
  int count = 0;
  VCHECK_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  VCHECK(count >= 2);
  int dev = -1;
  cudaDeviceProp want;
  std::memset(&want, 0, sizeof want);
  VCHECK_EQ(cudaChooseDevice(&dev, &want), cudaSuccess);   // asks for nothing: the first
  VCHECK_EQ(dev, 0);
  VCHECK_EQ(cudaGetDeviceProperties(&want, 1), cudaSuccess);
  VCHECK_EQ(cudaChooseDevice(&dev, &want), cudaSuccess);   // the devices are alike: the lowest
  VCHECK_EQ(dev, 0);
  want.major = 99;                                         // nothing has it; still an answer
  VCHECK_EQ(cudaChooseDevice(&dev, &want), cudaSuccess);
  VCHECK(dev >= 0 && dev < count);
  VCHECK_EQ(cudaChooseDevice(nullptr, &want), cudaErrorInvalidValue);
  VCHECK_EQ(cudaChooseDevice(&dev, nullptr), cudaErrorInvalidValue);
}

VTEST(init_device_starts_a_device_without_making_it_current) {
  VCHECK_EQ(cudaSetDevice(0), cudaSuccess);
  VCHECK_EQ(cudaInitDevice(1, 0, 0), cudaSuccess);
  int current = -1;
  VCHECK_EQ(cudaGetDevice(&current), cudaSuccess);
  VCHECK_EQ(current, 0);
  VCHECK_EQ(cudaInitDevice(1, 0, cudaInitDeviceFlagsAreValid), cudaSuccess);
  VCHECK_EQ(cudaInitDevice(99, 0, 0), cudaErrorInvalidDevice);
  VCHECK_EQ(cudaInitDevice(-1, 0, 0), cudaErrorInvalidDevice);
  VCHECK_EQ(cudaInitDevice(0, 0, 0x80), cudaErrorInvalidValue);   // an undefined flag
}

VTEST(a_pci_bus_id_names_the_device_it_came_from) {
  int count = 0;
  VCHECK_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  for (int d = 0; d < count; ++d) {
    char id[32];
    VCHECK_EQ(cudaDeviceGetPCIBusId(id, sizeof id, d), cudaSuccess);
    int back = -1;
    VCHECK_EQ(cudaDeviceGetByPCIBusId(&back, id), cudaSuccess);
    VCHECK_EQ(back, d);
    // Hexadecimal in either case, and the domain optional.
    std::string upper = id;
    for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    VCHECK_EQ(cudaDeviceGetByPCIBusId(&back, upper.c_str()), cudaSuccess);
    VCHECK_EQ(back, d);
    const std::string short_form = std::string(id).substr(5);   // drop "dddd:"
    VCHECK_EQ(cudaDeviceGetByPCIBusId(&back, short_form.c_str()), cudaSuccess);
    VCHECK_EQ(back, d);
  }
  int dev = -1;
  VCHECK_EQ(cudaDeviceGetByPCIBusId(&dev, "0000:ff:00.0"), cudaErrorInvalidDevice);   // well formed, no such bus
  VCHECK_EQ(cudaDeviceGetByPCIBusId(&dev, "not a bus id"), cudaErrorInvalidValue);
  VCHECK_EQ(cudaDeviceGetByPCIBusId(&dev, "0000:01:00.0 "), cudaErrorInvalidValue);   // trailing text
  VCHECK_EQ(cudaDeviceGetByPCIBusId(&dev, nullptr), cudaErrorInvalidValue);
  VCHECK_EQ(cudaDeviceGetByPCIBusId(nullptr, "0000:01:00.0"), cudaErrorInvalidValue);
}

// --- copies between arrays --------------------------------------------------

VTEST(2d_copies_into_and_out_of_arrays_have_async_forms_and_arrays_copy_to_arrays) {
  const cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uint32_t>();
  cudaArray_t a = nullptr, b = nullptr;
  VCHECK_EQ(cudaMallocArray(&a, &fmt, 8, 6), cudaSuccess);   // 8 texels of 4 bytes, 6 rows
  VCHECK_EQ(cudaMallocArray(&b, &fmt, 8, 6), cudaSuccess);
  cudaStream_t stream;
  VCHECK_EQ(cudaStreamCreate(&stream), cudaSuccess);

  std::vector<uint32_t> src(8 * 6), zero(8 * 6, 0);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint32_t>(1000 + i);
  VCHECK_EQ(cudaMemcpy2DToArrayAsync(a, 0, 0, src.data(), 32, 32, 6, cudaMemcpyHostToDevice, stream), cudaSuccess);
  VCHECK_EQ(cudaMemcpy2DToArrayAsync(b, 0, 0, zero.data(), 32, 32, 6, cudaMemcpyHostToDevice, stream), cudaSuccess);
  // A 3-texel by 2-row window of a, from (1, 2), into b at (4, 3).
  VCHECK_EQ(cudaMemcpy2DArrayToArray(b, 16, 3, a, 4, 2, 12, 2, cudaMemcpyDeviceToDevice), cudaSuccess);
  std::vector<uint32_t> out(8 * 6, 7);
  VCHECK_EQ(cudaMemcpy2DFromArrayAsync(out.data(), 32, b, 0, 0, 32, 6, cudaMemcpyDeviceToHost, stream), cudaSuccess);
  VCHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  for (size_t y = 0; y < 6; ++y)
    for (size_t x = 0; x < 8; ++x) {
      const bool inside = x >= 4 && x < 7 && y >= 3 && y < 5;
      const uint32_t want = inside ? src[(y - 3 + 2) * 8 + (x - 4 + 1)] : 0u;
      VCHECK_EQ(out[y * 8 + x], want);
    }

  // Out of range, wrong direction, not an array.
  VCHECK_EQ(cudaMemcpy2DArrayToArray(b, 16, 3, a, 4, 2, 20, 2, cudaMemcpyDeviceToDevice), cudaErrorInvalidValue);  // past the row
  VCHECK_EQ(cudaMemcpy2DArrayToArray(b, 0, 5, a, 0, 0, 8, 2, cudaMemcpyDeviceToDevice), cudaErrorInvalidValue);    // past the last row
  VCHECK_EQ(cudaMemcpy2DArrayToArray(b, 0, 0, a, 0, 0, 8, 2, cudaMemcpyHostToDevice), cudaErrorInvalidMemcpyDirection);
  VCHECK_EQ(cudaMemcpy2DArrayToArray(nullptr, 0, 0, a, 0, 0, 8, 2, cudaMemcpyDeviceToDevice), cudaErrorInvalidResourceHandle);
  VCHECK_EQ(cudaMemcpy2DArrayToArray(b, ~size_t{0}, 0, a, 0, 0, 8, 2, cudaMemcpyDeviceToDevice), cudaErrorInvalidValue);  // offset that would wrap
  VCHECK_EQ(cudaStreamDestroy(stream), cudaSuccess);
  VCHECK_EQ(cudaFreeArray(a), cudaSuccess);
  VCHECK_EQ(cudaFreeArray(b), cudaSuccess);
}

// --- texture and surface descriptors ----------------------------------------

VTEST(a_texture_object_hands_back_the_descriptors_it_was_made_from) {
  const cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
  cudaArray_t arr = nullptr;
  VCHECK_EQ(cudaMallocArray(&arr, &fmt, 16, 4), cudaSuccess);
  cudaResourceDesc res;
  std::memset(&res, 0, sizeof res);
  res.resType = cudaResourceTypeArray;
  res.res.array.array = arr;
  cudaTextureDesc td;
  std::memset(&td, 0, sizeof td);
  td.addressMode[0] = cudaAddressModeClamp;
  td.addressMode[1] = cudaAddressModeWrap;
  td.filterMode = cudaFilterModeLinear;
  td.readMode = cudaReadModeElementType;
  td.normalizedCoords = 1;
  cudaTextureObject_t tex = 0;
  VCHECK_EQ(cudaCreateTextureObject(&tex, &res, &td, nullptr), cudaSuccess);

  cudaResourceDesc got_res;
  VCHECK_EQ(cudaGetTextureObjectResourceDesc(&got_res, tex), cudaSuccess);
  VCHECK_EQ(static_cast<int>(got_res.resType), static_cast<int>(cudaResourceTypeArray));
  VCHECK(got_res.res.array.array == arr);
  cudaTextureDesc got_tex;
  VCHECK_EQ(cudaGetTextureObjectTextureDesc(&got_tex, tex), cudaSuccess);
  VCHECK_EQ(static_cast<int>(got_tex.addressMode[1]), static_cast<int>(cudaAddressModeWrap));
  VCHECK_EQ(static_cast<int>(got_tex.filterMode), static_cast<int>(cudaFilterModeLinear));
  VCHECK_EQ(got_tex.normalizedCoords, 1);
  cudaResourceViewDesc view;
  VCHECK_EQ(cudaGetTextureObjectResourceViewDesc(&view, tex), cudaErrorInvalidValue);   // none was set

  VCHECK_EQ(cudaGetTextureObjectResourceDesc(nullptr, tex), cudaErrorInvalidValue);
  VCHECK_EQ(cudaDestroyTextureObject(tex), cudaSuccess);
  VCHECK_EQ(cudaGetTextureObjectResourceDesc(&got_res, tex), cudaErrorInvalidValue);   // gone
  VCHECK_EQ(cudaGetTextureObjectTextureDesc(&got_tex, tex), cudaErrorInvalidValue);
  VCHECK_EQ(cudaFreeArray(arr), cudaSuccess);
}

VTEST(a_surface_object_hands_back_its_resource_descriptor) {
  const cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uint32_t>();
  cudaArray_t arr = nullptr;
  VCHECK_EQ(cudaMallocArray(&arr, &fmt, 8, 8, cudaArraySurfaceLoadStore), cudaSuccess);
  cudaResourceDesc res;
  std::memset(&res, 0, sizeof res);
  res.resType = cudaResourceTypeArray;
  res.res.array.array = arr;
  cudaSurfaceObject_t surf = 0;
  VCHECK_EQ(cudaCreateSurfaceObject(&surf, &res), cudaSuccess);
  cudaResourceDesc got;
  VCHECK_EQ(cudaGetSurfaceObjectResourceDesc(&got, surf), cudaSuccess);
  VCHECK(got.res.array.array == arr);
  VCHECK_EQ(cudaDestroySurfaceObject(surf), cudaSuccess);
  VCHECK_EQ(cudaGetSurfaceObjectResourceDesc(&got, surf), cudaErrorInvalidValue);
  VCHECK_EQ(cudaFreeArray(arr), cudaSuccess);
}

// --- memory advice with a location ------------------------------------------

#if CUDART_VERSION >= 12020 && CUDART_VERSION < 13000
VTEST(advice_and_prefetch_take_a_location_under_their_v2_names) {
  void* p = nullptr;
  VCHECK_EQ(cudaMallocManaged(&p, 1 << 16), cudaSuccess);
  cudaMemLocation dev0{};
  dev0.type = cudaMemLocationTypeDevice;
  dev0.id = 0;
  VCHECK_EQ(cudaMemAdvise_v2(p, 1 << 16, cudaMemAdviseSetPreferredLocation, dev0), cudaSuccess);
  int preferred = -2;
  VCHECK_EQ(cudaMemRangeGetAttribute(&preferred, sizeof preferred, cudaMemRangeAttributePreferredLocation, p, 1 << 16),
            cudaSuccess);
  VCHECK_EQ(preferred, 0);
  cudaMemLocation host{};
  host.type = cudaMemLocationTypeHost;
  VCHECK_EQ(cudaMemAdvise_v2(p, 1 << 16, cudaMemAdviseSetPreferredLocation, host), cudaSuccess);
  VCHECK_EQ(cudaMemRangeGetAttribute(&preferred, sizeof preferred, cudaMemRangeAttributePreferredLocation, p, 1 << 16),
            cudaSuccess);
  VCHECK_EQ(preferred, cudaCpuDeviceId);
  VCHECK_EQ(cudaMemPrefetchAsync_v2(p, 1 << 16, dev0, 0, nullptr), cudaSuccess);
  int last = -2;
  VCHECK_EQ(cudaMemRangeGetAttribute(&last, sizeof last, cudaMemRangeAttributeLastPrefetchLocation, p, 1 << 16),
            cudaSuccess);
  VCHECK_EQ(last, 0);
  cudaMemLocation bad{};
  bad.type = cudaMemLocationTypeInvalid;
  VCHECK_EQ(cudaMemAdvise_v2(p, 1 << 16, cudaMemAdviseSetReadMostly, bad), cudaErrorInvalidValue);
  VCHECK_EQ(cudaMemPrefetchAsync_v2(p, 1 << 16, bad, 0, nullptr), cudaErrorInvalidValue);
  VCHECK_EQ(cudaFree(p), cudaSuccess);
}
#endif

// --- a registered kernel's name and parameters ------------------------------

#if CUDART_VERSION >= 12040
const char kKernelPtx[] = R"(
.version 7.0
.target sm_80
.address_size 64
.visible .entry _Z5shapePfifP4blobh(
  .param .u64 p,
  .param .u32 n,
  .param .f32 v,
  .param .align 16 .b8 blob[24],
  .param .u8 c
)
{
  ret;
}
)";

struct Wrapper {   // __fatBinC_Wrapper_t
  int magic;
  int version;
  const void* data;
  const void* filename;
};

void stub() {}

VTEST(a_registered_kernel_reports_its_name_and_where_its_parameters_sit) {
  vgpu::cuda::FatbinImage image;
  image.kind = vgpu::cuda::kFatbinPtx;
  image.arch = 80;
  image.major = 7;
  image.minor = 0;
  image.data = kKernelPtx;
  static const std::string fatbin = vgpu::cuda::write_fatbin({image});
  static Wrapper wrapper{0x466243B1, 1, fatbin.data(), nullptr};
  void** handle = __cudaRegisterFatBinary(&wrapper);
  VCHECK(handle != nullptr);
  __cudaRegisterFunction(handle, reinterpret_cast<const char*>(&stub), const_cast<char*>("_Z5shapePfifP4blobh"),
                         "_Z5shapePfifP4blobh", -1, nullptr, nullptr, nullptr, nullptr, nullptr);

  const char* name = nullptr;
  VCHECK_EQ(cudaFuncGetName(&name, reinterpret_cast<const void*>(&stub)), cudaSuccess);
  VCHECK_EQ(std::string(name), std::string("_Z5shapePfifP4blobh"));
  VCHECK_EQ(cudaFuncGetName(nullptr, reinterpret_cast<const void*>(&stub)), cudaErrorInvalidValue);
  VCHECK_EQ(cudaFuncGetName(&name, reinterpret_cast<const void*>(&kKernelPtx)), cudaErrorInvalidDeviceFunction);

  // Each parameter at the next offset its alignment allows: the pointer at 0,
  // the int at 8, the float at 12, the 16-aligned 24-byte structure at 16, the
  // byte at 40.
  const size_t want_offset[] = {0, 8, 12, 16, 40};
  const size_t want_size[] = {8, 4, 4, 24, 1};
  for (size_t i = 0; i < 5; ++i) {
    size_t offset = ~size_t{0}, size = 0;
    VCHECK_EQ(cudaFuncGetParamInfo(reinterpret_cast<const void*>(&stub), i, &offset, &size), cudaSuccess);
    VCHECK_EQ(offset, want_offset[i]);
    VCHECK_EQ(size, want_size[i]);
  }
  size_t offset = 0;
  VCHECK_EQ(cudaFuncGetParamInfo(reinterpret_cast<const void*>(&stub), 2, &offset, nullptr), cudaSuccess);   // size is optional
  VCHECK_EQ(offset, size_t{12});
  VCHECK_EQ(cudaFuncGetParamInfo(reinterpret_cast<const void*>(&stub), 5, &offset, nullptr), cudaErrorInvalidValue);
  VCHECK_EQ(cudaFuncGetParamInfo(reinterpret_cast<const void*>(&stub), 0, nullptr, nullptr), cudaErrorInvalidValue);
}
#endif

// --- what cluster occupancy needs and no profile has ------------------------

VTEST(cluster_occupancy_is_refused_by_name_not_answered_from_a_made_up_layout) {
  cudaLaunchConfig_t cfg{};
  int n = 0;
  static const char probe = 0;   // any address: the refusal does not look the function up (kKernelPtx exists only from 12.4)
  VCHECK_EQ(cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(&probe), &cfg), cudaErrorNotSupported);
  VCHECK_EQ(cudaOccupancyMaxPotentialClusterSize(&n, reinterpret_cast<const void*>(&probe), &cfg), cudaErrorNotSupported);
  VCHECK_EQ(cudaOccupancyMaxActiveClusters(nullptr, nullptr, &cfg), cudaErrorInvalidValue);
}

// --- deprecated spellings ---------------------------------------------------

#if CUDART_VERSION < 13000
VTEST(shared_memory_banks_are_four_bytes_wide) {
  cudaSharedMemConfig c = cudaSharedMemBankSizeEightByte;
  VCHECK_EQ(cudaDeviceGetSharedMemConfig(&c), cudaSuccess);
  VCHECK_EQ(static_cast<int>(c), static_cast<int>(cudaSharedMemBankSizeFourByte));
  VCHECK_EQ(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte), cudaSuccess);   // accepted, no effect
  VCHECK_EQ(cudaDeviceGetSharedMemConfig(&c), cudaSuccess);
  VCHECK_EQ(static_cast<int>(c), static_cast<int>(cudaSharedMemBankSizeFourByte));
  // An out-of-range enumerator, made without the cast that would itself be UB.
  const int bad = 7;
  cudaSharedMemConfig seven;
  std::memcpy(&seven, &bad, sizeof seven);
  VCHECK_EQ(cudaDeviceSetSharedMemConfig(seven), cudaErrorInvalidValue);
  VCHECK_EQ(cudaDeviceGetSharedMemConfig(nullptr), cudaErrorInvalidValue);
}

VTEST(the_cudaThread_calls_are_the_cudaDevice_calls) {
  VCHECK_EQ(cudaThreadSetLimit(cudaLimitMallocHeapSize, 32u << 20), cudaSuccess);
  size_t v = 0;
  VCHECK_EQ(cudaDeviceGetLimit(&v, cudaLimitMallocHeapSize), cudaSuccess);
  VCHECK_EQ(v, size_t{32u << 20});
  v = 0;
  VCHECK_EQ(cudaThreadGetLimit(&v, cudaLimitMallocHeapSize), cudaSuccess);
  VCHECK_EQ(v, size_t{32u << 20});
  VCHECK_EQ(cudaThreadSetCacheConfig(cudaFuncCachePreferShared), cudaSuccess);
  cudaFuncCache cache = cudaFuncCachePreferNone;
  VCHECK_EQ(cudaDeviceGetCacheConfig(&cache), cudaSuccess);
  VCHECK_EQ(static_cast<int>(cache), static_cast<int>(cudaFuncCachePreferShared));
  cache = cudaFuncCachePreferNone;
  VCHECK_EQ(cudaThreadGetCacheConfig(&cache), cudaSuccess);
  VCHECK_EQ(static_cast<int>(cache), static_cast<int>(cudaFuncCachePreferShared));
  VCHECK_EQ(cudaThreadGetLimit(nullptr, cudaLimitMallocHeapSize), cudaErrorInvalidValue);
}

// Last: it resets the device.
VTEST(zz_thread_exit_resets_the_device) {
  void* p = nullptr;
  VCHECK_EQ(cudaMalloc(&p, 4096), cudaSuccess);
  VCHECK_EQ(cudaThreadExit(), cudaSuccess);
  VCHECK_EQ(cudaFree(p), cudaErrorInvalidValue);   // the allocation went with the reset
}
#endif

}  // namespace

VTEST_MAIN
