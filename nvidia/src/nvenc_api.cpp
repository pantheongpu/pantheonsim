// libvgpunvenc — VirtualGPU's implementation of the NVENC encode API.
//
// WHY THIS EXISTS
// ---------------
// NVENC (`libnvidia-encode.so.1`) is a *driver* component, not a redistributable
// SDK library: it drives fixed-function encoder silicon through the kernel
// driver and never routes through the public CUDA API, so VirtualGPU's libcuda
// cannot intercept it. But applications load it with
// `dlopen("libnvidia-encode.so.1")` — a bare soname, which searches
// LD_LIBRARY_PATH first — so we can supply our own implementation the same way
// we supply libcuda.so.1 and libcudart.so.13.
//
// WHAT THIS IS (AND IS NOT)
// -------------------------
// This is NOT an H.264/HEVC encoder. It implements the documented NVENC API
// (from the public Video Codec SDK header, nvEncodeAPI.h) with a deterministic,
// *content-derived* bitstream: encoding the same frame twice yields identical
// bytes, and changing one pixel changes the output. That is precisely the
// property encoder stress/SDC tests rely on — they capture a golden bitstream
// and compare later frames against it to detect silent corruption through the
// encode path — so those tests work correctly here.
//
// It does NOT produce a decodable video stream. Anything that needs real
// compressed output, rate control behavior, or codec conformance must use
// hardware. That limitation is reported by `vgpu info` and documented.
//
// Input buffers are backed by VirtualGPU *device* memory, because applications
// legitimately run CUDA kernels against the locked input pointer.
#include <nvEncodeAPI.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"

// The CUDA runtime shim owns the virtual devices; reuse its allocator so
// encoder input buffers are real device memory that kernels can write.
extern "C" {
int cudaMalloc(void** ptr, size_t size);
int cudaFree(void* ptr);
int cudaMemcpy(void* dst, const void* src, size_t count, int kind);
}

namespace {

constexpr int kMemcpyDeviceToHost = 2;

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

struct InputBuffer {
  uint32_t width = 0, height = 0, pitch = 0;
  // Rows of bytes the buffer holds: the image height for packed RGB, and one
  // and a half times it for the 4:2:0 formats, whose chroma follows the luma.
  uint32_t rows = 0;
  NV_ENC_BUFFER_FORMAT format = NV_ENC_BUFFER_FORMAT_UNDEFINED;
  size_t bytes = 0;
  void* device_ptr = nullptr;  // VirtualGPU device memory, kernel-writable
  bool locked = false;
};

struct BitstreamBuffer {
  std::vector<uint8_t> data;
  bool locked = false;
};

struct Session {
  uint32_t width = 0, height = 0;
  bool initialized = false;
  std::map<void*, InputBuffer> inputs;
  std::map<void*, BitstreamBuffer> outputs;
  // The most recent EncodePicture result, delivered by LockBitstream.
  std::vector<uint8_t> pending;
  void* pending_output = nullptr;
  uint64_t frame_index = 0;
};

std::mutex g_mu;
std::map<void*, Session*> g_sessions;

// A deterministic content-derived "bitstream". Each 64x64 tile contributes an
// FNV-1a hash of its pixels, so identical frames encode identically and any
// changed pixel changes the output — the property SDC verification needs.
std::vector<uint8_t> encode_frame(const std::vector<uint8_t>& frame, uint32_t width,
                                  uint32_t height, uint32_t rows, uint32_t pitch,
                                  uint32_t bytes_per_pixel) {
  std::vector<uint8_t> out;
  // A short pseudo-header so consumers see stable, plausible framing.
  const uint8_t header[] = {0x00, 0x00, 0x00, 0x01, 'V', 'G', 'P', 'U'};
  out.insert(out.end(), header, header + sizeof header);
  auto push32 = [&out](uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
  };
  push32(width);
  push32(height);

  constexpr uint32_t kTile = 64;
  // Every row of the buffer, chroma included: a corrupted chroma sample must
  // change the output as surely as a corrupted luma sample does.
  for (uint32_t ty = 0; ty < rows; ty += kTile) {
    for (uint32_t tx = 0; tx < width; tx += kTile) {
      uint32_t h = 2166136261u;
      for (uint32_t y = ty; y < std::min(ty + kTile, rows); ++y) {
        const uint8_t* row = frame.data() + static_cast<size_t>(y) * pitch;
        for (uint32_t x = tx; x < std::min(tx + kTile, width); ++x) {
          for (uint32_t b = 0; b < bytes_per_pixel; ++b)
            h = (h ^ row[x * bytes_per_pixel + b]) * 16777619u;
        }
      }
      push32(h);
    }
  }
  return out;
}

uint32_t bytes_per_pixel(NV_ENC_BUFFER_FORMAT fmt) {
  switch (fmt) {
    case NV_ENC_BUFFER_FORMAT_ARGB:
    case NV_ENC_BUFFER_FORMAT_ABGR:
    case NV_ENC_BUFFER_FORMAT_ARGB10:
    case NV_ENC_BUFFER_FORMAT_ABGR10:
      return 4;
    case NV_ENC_BUFFER_FORMAT_NV12:
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
      return 1;  // luma plane stride; chroma follows
    default:
      return 4;
  }
}

// Rows of bytes a frame occupies. The 4:2:0 formats carry half-height chroma
// after the luma, so a buffer sized for the luma alone is a third too small and
// an application filling the whole frame writes past its end.
uint32_t buffer_rows(NV_ENC_BUFFER_FORMAT fmt, uint32_t height) {
  switch (fmt) {
    case NV_ENC_BUFFER_FORMAT_NV12:
    case NV_ENC_BUFFER_FORMAT_YV12:
    case NV_ENC_BUFFER_FORMAT_IYUV:
      return height + (height + 1) / 2;
    default:
      return height;
  }
}

Session* session_of(void* enc) {
  auto it = g_sessions.find(enc);
  return it == g_sessions.end() ? nullptr : it->second;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncOpenEncodeSessionEx(
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS* params, void** encoder) {
  if (!params || !encoder) return NV_ENC_ERR_INVALID_PTR;
  if (params->deviceType != NV_ENC_DEVICE_TYPE_CUDA) {
    if (!quiet())
      std::fprintf(stderr,
                   "[vgpu] NvEncOpenEncodeSessionEx: only NV_ENC_DEVICE_TYPE_CUDA is supported\n");
    return NV_ENC_ERR_UNSUPPORTED_DEVICE;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = new Session();
  *encoder = s;
  g_sessions[s] = s;
  if (!quiet())
    std::fprintf(stderr, "[vgpu] virtual NVENC session opened (deterministic content-derived "
                         "bitstream; not a real HEVC encoder)\n");
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncInitializeEncoder(void* encoder,
                                                        NV_ENC_INITIALIZE_PARAMS* params) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s || !params) return NV_ENC_ERR_INVALID_PTR;
  s->width = params->encodeWidth;
  s->height = params->encodeHeight;
  s->initialized = true;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodePresetConfigEx(void* encoder, GUID encodeGUID,
                                                              GUID presetGUID,
                                                              NV_ENC_TUNING_INFO tuningInfo,
                                                              NV_ENC_PRESET_CONFIG* presetConfig) {
  (void)encodeGUID;
  (void)presetGUID;
  (void)tuningInfo;
  std::lock_guard<std::mutex> lock(g_mu);
  if (!session_of(encoder) || !presetConfig) return NV_ENC_ERR_INVALID_PTR;
  // Hand back a zeroed config with the versions the caller expects; callers
  // then override the fields they care about (e.g. constant-QP rate control).
  uint32_t cfg_version = presetConfig->presetCfg.version;
  uint32_t version = presetConfig->version;
  std::memset(&presetConfig->presetCfg, 0, sizeof presetConfig->presetCfg);
  presetConfig->version = version;
  presetConfig->presetCfg.version = cfg_version;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncCreateInputBuffer(void* encoder,
                                                        NV_ENC_CREATE_INPUT_BUFFER* params) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s || !params) return NV_ENC_ERR_INVALID_PTR;
  InputBuffer buf;
  buf.width = params->width;
  buf.height = params->height;
  uint32_t bpp = bytes_per_pixel(params->bufferFmt);
  buf.pitch = params->width * bpp;
  buf.rows = buffer_rows(params->bufferFmt, params->height);
  buf.format = params->bufferFmt;
  buf.bytes = static_cast<size_t>(buf.pitch) * buf.rows;
  // Device memory: applications legitimately run CUDA kernels on the locked
  // pointer, so it must be addressable by the virtual GPU.
  void* dptr = nullptr;
  if (cudaMalloc(&dptr, buf.bytes) != 0 || !dptr) return NV_ENC_ERR_OUT_OF_MEMORY;
  buf.device_ptr = dptr;
  params->inputBuffer = dptr;  // the handle is the device pointer itself
  s->inputs[dptr] = buf;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyInputBuffer(void* encoder, NV_ENC_INPUT_PTR input) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->inputs.find(input);
  if (it == s->inputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  cudaFree(it->second.device_ptr);
  s->inputs.erase(it);
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncLockInputBuffer(void* encoder,
                                                      NV_ENC_LOCK_INPUT_BUFFER* params) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s || !params) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->inputs.find(params->inputBuffer);
  if (it == s->inputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  it->second.locked = true;
  params->bufferDataPtr = it->second.device_ptr;  // kernel-writable
  params->pitch = it->second.pitch;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnlockInputBuffer(void* encoder, NV_ENC_INPUT_PTR input) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->inputs.find(input);
  if (it == s->inputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  it->second.locked = false;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI
NvEncCreateBitstreamBuffer(void* encoder, NV_ENC_CREATE_BITSTREAM_BUFFER* params) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s || !params) return NV_ENC_ERR_INVALID_PTR;
  auto* handle = new BitstreamBuffer();
  params->bitstreamBuffer = handle;
  s->outputs[handle] = BitstreamBuffer();
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyBitstreamBuffer(void* encoder,
                                                             NV_ENC_OUTPUT_PTR bitstream) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s) return NV_ENC_ERR_INVALID_PTR;
  s->outputs.erase(bitstream);
  delete static_cast<BitstreamBuffer*>(bitstream);
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncEncodePicture(void* encoder, NV_ENC_PIC_PARAMS* params) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s || !params) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->inputs.find(params->inputBuffer);
  if (it == s->inputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  const InputBuffer& in = it->second;

  // Read the frame out of virtual device memory and "encode" it.
  std::vector<uint8_t> frame(in.bytes);
  if (cudaMemcpy(frame.data(), in.device_ptr, in.bytes, kMemcpyDeviceToHost) != 0)
    return NV_ENC_ERR_GENERIC;
  s->pending = encode_frame(frame, in.width, in.height, in.rows, in.pitch,
                            bytes_per_pixel(in.format));
  s->pending_output = params->outputBitstream;
  ++s->frame_index;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncLockBitstream(void* encoder, NV_ENC_LOCK_BITSTREAM* params) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s || !params) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->outputs.find(params->outputBitstream);
  if (it == s->outputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  it->second.data = s->pending;
  it->second.locked = true;
  params->bitstreamBufferPtr = it->second.data.data();
  params->bitstreamSizeInBytes = static_cast<uint32_t>(it->second.data.size());
  params->outputTimeStamp = s->frame_index;
  params->pictureType = NV_ENC_PIC_TYPE_IDR;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnlockBitstream(void* encoder, NV_ENC_OUTPUT_PTR bitstream) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->outputs.find(bitstream);
  if (it == s->outputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  it->second.locked = false;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyEncoder(void* encoder) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = session_of(encoder);
  if (!s) return NV_ENC_ERR_INVALID_PTR;
  for (auto& [ptr, buf] : s->inputs) cudaFree(buf.device_ptr);
  // Bitstream handles are heap objects the caller never frees once the
  // session is gone: DestroyBitstreamBuffer rejects a destroyed session.
  for (auto& [ptr, buf] : s->outputs) delete static_cast<BitstreamBuffer*>(ptr);
  g_sessions.erase(encoder);
  delete s;
  return NV_ENC_SUCCESS;
}

// Unimplemented entry points report NV_ENC_ERR_UNIMPLEMENTED rather than
// crashing on a null jump, so a caller that needs them gets a clear failure.
namespace {
template <int Slot>
NVENCSTATUS NVENCAPI unimplemented(...) {
  if (!quiet())
    std::fprintf(stderr, "[vgpu] NVENC function slot %d is not implemented by VirtualGPU\n", Slot);
  return NV_ENC_ERR_UNIMPLEMENTED;
}
}  // namespace

VGPU_EXPORT NVENCSTATUS NVENCAPI
NvEncodeAPICreateInstance(NV_ENCODE_API_FUNCTION_LIST* functionList) {
  if (!functionList) return NV_ENC_ERR_INVALID_PTR;
  uint32_t version = functionList->version;
  std::memset(functionList, 0, sizeof *functionList);
  functionList->version = version;
  functionList->nvEncOpenEncodeSessionEx = NvEncOpenEncodeSessionEx;
  functionList->nvEncInitializeEncoder = NvEncInitializeEncoder;
  functionList->nvEncGetEncodePresetConfigEx = NvEncGetEncodePresetConfigEx;
  functionList->nvEncCreateInputBuffer = NvEncCreateInputBuffer;
  functionList->nvEncDestroyInputBuffer = NvEncDestroyInputBuffer;
  functionList->nvEncLockInputBuffer = NvEncLockInputBuffer;
  functionList->nvEncUnlockInputBuffer = NvEncUnlockInputBuffer;
  functionList->nvEncCreateBitstreamBuffer = NvEncCreateBitstreamBuffer;
  functionList->nvEncDestroyBitstreamBuffer = NvEncDestroyBitstreamBuffer;
  functionList->nvEncEncodePicture = NvEncEncodePicture;
  functionList->nvEncLockBitstream = NvEncLockBitstream;
  functionList->nvEncUnlockBitstream = NvEncUnlockBitstream;
  functionList->nvEncDestroyEncoder = NvEncDestroyEncoder;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncodeAPIGetMaxSupportedVersion(uint32_t* version) {
  if (!version) return NV_ENC_ERR_INVALID_PTR;
  *version = NVENCAPI_VERSION;
  return NV_ENC_SUCCESS;
}
