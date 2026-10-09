// NVENC through the documented API, loaded by the bare soname applications
// dlopen, compiled with nvcc against the public Video Codec SDK header.
//
// What encoder stress and corruption checks rely on: the same frame encodes to
// the same bytes, and a changed byte changes them. For the 4:2:0 frame the
// changed byte is a chroma sample, and the whole frame, luma and chroma, is
// written (here by a kernel) into the buffer the encoder hands out. That buffer
// is managed memory in the simulator; nvenc_h264.cpp decodes the stream.
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvEncodeAPI.h>

// The frame the pantheon media_enc_virus workload builds (an ARGB gradient), and the
// pixel it corrupts to see whether the encoder notices.
__global__ void gradient(unsigned char* ptr, int width, int height, int pitch) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = idx; i < (size_t)width * height; i += stride) {
    const int x = (int)(i % width), y = (int)(i / width);
    *(uint32_t*)(ptr + (size_t)y * pitch + (size_t)x * 4) = 0xFF000000u | ((x & 0xFF) << 16) | ((y & 0xFF) << 8) | ((x ^ y) & 0xFF);
  }
}
__global__ void corrupt(unsigned char* ptr, int pitch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *(uint32_t*)(ptr + 337 * (size_t)pitch + 337 * 4) ^= 0x00FFFFFFu;
}

__global__ void fill(unsigned char* p, size_t n, unsigned char seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = (unsigned char)(i * 31u + seed);
}

namespace {

int failures = 0;

void fail(const char* format, const char* name, const char* detail = "") {
  std::printf(format, name, detail);
  std::printf("\n");
  ++failures;
}

struct Encoder {
  NV_ENCODE_API_FUNCTION_LIST api{};
  void* session = nullptr;
};

std::vector<unsigned char> encode(Encoder& e, NV_ENC_INPUT_PTR in, NV_ENC_OUTPUT_PTR out,
                                  NV_ENC_BUFFER_FORMAT fmt, uint32_t width, uint32_t height, uint32_t flags = 0) {
  NV_ENC_PIC_PARAMS pic{};
  pic.version = NV_ENC_PIC_PARAMS_VER;
  pic.inputBuffer = in;
  pic.outputBitstream = out;
  pic.bufferFmt = fmt;
  pic.inputWidth = width;
  pic.inputHeight = height;
  pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  pic.encodePicFlags = flags;
  if (e.api.nvEncEncodePicture(e.session, &pic) != NV_ENC_SUCCESS) return {};
  NV_ENC_LOCK_BITSTREAM lock{};
  lock.version = NV_ENC_LOCK_BITSTREAM_VER;
  lock.outputBitstream = out;
  if (e.api.nvEncLockBitstream(e.session, &lock) != NV_ENC_SUCCESS) return {};
  const auto* b = static_cast<const unsigned char*>(lock.bitstreamBufferPtr);
  std::vector<unsigned char> bytes(b, b + lock.bitstreamSizeInBytes);
  e.api.nvEncUnlockBitstream(e.session, out);
  return bytes;
}

// One buffer format end to end. `rows` is how many rows of `pitch` bytes the
// frame occupies: the height for packed RGB, one and a half times it for 4:2:0.
void check_format(Encoder& e, NV_ENC_BUFFER_FORMAT fmt, const char* name, uint32_t width,
                  uint32_t height, uint32_t rows) {
  NV_ENC_CREATE_INPUT_BUFFER input{};
  input.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
  input.width = width;
  input.height = height;
  input.bufferFmt = fmt;
  NV_ENC_CREATE_BITSTREAM_BUFFER output{};
  output.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
  if (e.api.nvEncCreateInputBuffer(e.session, &input) != NV_ENC_SUCCESS) {
    fail("FAIL %s: could not create the input buffer%s", name);
    return;
  }
  if (e.api.nvEncCreateBitstreamBuffer(e.session, &output) != NV_ENC_SUCCESS) {
    fail("FAIL %s: could not create the bitstream buffer%s", name);
    e.api.nvEncDestroyInputBuffer(e.session, input.inputBuffer);
    return;
  }
  auto done = [&] {
    e.api.nvEncDestroyBitstreamBuffer(e.session, output.bitstreamBuffer);
    e.api.nvEncDestroyInputBuffer(e.session, input.inputBuffer);
  };

  NV_ENC_LOCK_INPUT_BUFFER lock{};
  lock.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
  lock.inputBuffer = input.inputBuffer;
  if (e.api.nvEncLockInputBuffer(e.session, &lock) != NV_ENC_SUCCESS) {
    fail("FAIL %s: could not lock the input buffer%s", name);
    done();
    return;
  }
  auto* frame = static_cast<unsigned char*>(lock.bufferDataPtr);
  const size_t n = static_cast<size_t>(lock.pitch) * rows;
  fill<<<(unsigned)((n + 255) / 256), 256>>>(frame, n, 7);
  cudaError_t err = cudaDeviceSynchronize();
  e.api.nvEncUnlockInputBuffer(e.session, input.inputBuffer);
  if (err != cudaSuccess) {
    fail("FAIL %s: filling the whole frame: %s", name, cudaGetErrorString(err));
    done();
    return;
  }

  // Consecutive IDR pictures differ in idr_pic_id, and the first one is preceded by
  // the parameter sets, so "the same frame" is compared two pictures apart and
  // after those headers.
  const auto first = encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height);
  const auto again = encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height);
  const auto third = encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height);
  if (first.empty()) fail("FAIL %s: encoding produced no bytes%s", name);
  if (third.empty() || third.size() > first.size() ||
      !std::equal(third.begin(), third.end(), first.end() - static_cast<std::ptrdiff_t>(third.size())))
    fail("FAIL %s: the same frame encoded differently%s", name);

  // The last byte of an ARGB frame is the last pixel's alpha, which the encoder
  // ignores; its red is the byte before.
  const size_t victim = fmt == NV_ENC_BUFFER_FORMAT_ARGB ? n - 2 : n - 1;
  unsigned char last = 0;
  cudaMemcpy(&last, frame + victim, 1, cudaMemcpyDeviceToHost);
  last ^= 0x5a;
  cudaMemcpy(frame + victim, &last, 1, cudaMemcpyHostToDevice);
  const auto changed = encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height);
  if (changed.empty() || changed == again)
    fail("FAIL %s: changing the frame's last byte did not change the output%s", name);
  // The pantheon media_enc_virus contract: with a forced IDR and the parameter sets every
  // time, every encode of one frame is the same bytes, and a changed byte is not.
  const uint32_t forced = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
  std::vector<unsigned char> golden = encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height, forced);
  for (int i = 0; i < 3; ++i) {
    const auto again2 = encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height, forced);
    if (again2 != golden || golden.empty()) fail("FAIL %s: a forced-IDR encode of the same frame differed from the golden one%s", name);
  }
  last ^= 0x5a;   // back to the original frame
  cudaMemcpy(frame + victim, &last, 1, cudaMemcpyHostToDevice);
  if (encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height, forced) == golden)
    fail("FAIL %s: a changed frame encoded to the golden bytes%s", name);
  done();
}

// What media_enc_virus does, at 1280x720: HEVC, preset P7 with its high-quality tuning and a
// constant QP, an ARGB input buffer filled by a kernel, every frame a forced IDR with the
// parameter sets; one golden frame, then the same frame again and again, then the frame with
// one pixel corrupted. The workload passes when the first repeats are the golden bytes and
// the corrupted one is not.
void workload_contract(Encoder& e, const NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS& open) {
  const char* name = "media_enc_virus";
  void* session = nullptr;
  if (e.api.nvEncOpenEncodeSessionEx(const_cast<NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS*>(&open), &session) != NV_ENC_SUCCESS) {
    fail("FAIL %s: no session%s", name);
    return;
  }
  Encoder w;
  w.api = e.api;
  w.session = session;
  const uint32_t W = 1280, H = 720;
  NV_ENC_INITIALIZE_PARAMS ip{};
  ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
  ip.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
  ip.presetGUID = NV_ENC_PRESET_P7_GUID;
  ip.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
  ip.encodeWidth = ip.maxEncodeWidth = ip.darWidth = W;
  ip.encodeHeight = ip.maxEncodeHeight = ip.darHeight = H;
  ip.frameRateNum = 60;
  ip.frameRateDen = 1;
  ip.enablePTD = 1;
  NV_ENC_PRESET_CONFIG pc{};
  pc.version = NV_ENC_PRESET_CONFIG_VER;
  pc.presetCfg.version = NV_ENC_CONFIG_VER;
  if (e.api.nvEncGetEncodePresetConfigEx(session, ip.encodeGUID, ip.presetGUID, NV_ENC_TUNING_INFO_HIGH_QUALITY, &pc) != NV_ENC_SUCCESS) {
    fail("FAIL %s: preset configuration refused%s", name);
    return;
  }
  pc.presetCfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
  pc.presetCfg.rcParams.constQP.qpInterP = pc.presetCfg.rcParams.constQP.qpInterB = pc.presetCfg.rcParams.constQP.qpIntra = 28;
  ip.encodeConfig = &pc.presetCfg;
  if (e.api.nvEncInitializeEncoder(session, &ip) != NV_ENC_SUCCESS) {
    fail("FAIL %s: initialisation refused%s", name);
    return;
  }
  NV_ENC_CREATE_INPUT_BUFFER ib{};
  ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
  ib.width = W;
  ib.height = H;
  ib.bufferFmt = NV_ENC_BUFFER_FORMAT_ARGB;
  NV_ENC_CREATE_BITSTREAM_BUFFER ob{};
  ob.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
  if (e.api.nvEncCreateInputBuffer(session, &ib) != NV_ENC_SUCCESS || e.api.nvEncCreateBitstreamBuffer(session, &ob) != NV_ENC_SUCCESS) {
    fail("FAIL %s: buffers refused%s", name);
    return;
  }
  NV_ENC_LOCK_INPUT_BUFFER lk{};
  lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
  lk.inputBuffer = ib.inputBuffer;
  e.api.nvEncLockInputBuffer(session, &lk);
  gradient<<<64, 256>>>((unsigned char*)lk.bufferDataPtr, (int)W, (int)H, (int)lk.pitch);
  cudaDeviceSynchronize();
  e.api.nvEncUnlockInputBuffer(session, ib.inputBuffer);
  const uint32_t forced = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
  const auto golden = encode(w, ib.inputBuffer, ob.bitstreamBuffer, NV_ENC_BUFFER_FORMAT_ARGB, W, H, forced);
  bool same = !golden.empty();
  for (int i = 0; i < 5; ++i) same = same && encode(w, ib.inputBuffer, ob.bitstreamBuffer, NV_ENC_BUFFER_FORMAT_ARGB, W, H, forced) == golden;
  if (!same) fail("FAIL %s: a repeat of the golden frame differed from it%s", name);
  e.api.nvEncLockInputBuffer(session, &lk);
  corrupt<<<1, 1>>>((unsigned char*)lk.bufferDataPtr, (int)lk.pitch);
  cudaDeviceSynchronize();
  e.api.nvEncUnlockInputBuffer(session, ib.inputBuffer);
  if (encode(w, ib.inputBuffer, ob.bitstreamBuffer, NV_ENC_BUFFER_FORMAT_ARGB, W, H, forced) == golden)
    fail("FAIL %s: a corrupted pixel did not change the bitstream%s", name);
  e.api.nvEncDestroyBitstreamBuffer(session, ob.bitstreamBuffer);
  e.api.nvEncDestroyInputBuffer(session, ib.inputBuffer);
  e.api.nvEncDestroyEncoder(session);
}

}  // namespace

int main() {
  void* lib = dlopen("libnvidia-encode.so.1", RTLD_NOW);
  if (!lib) {
    std::printf("FAIL dlopen libnvidia-encode.so.1: %s\n", dlerror());
    return 1;
  }
  using Create = NVENCSTATUS (*)(NV_ENCODE_API_FUNCTION_LIST*);
  auto create = reinterpret_cast<Create>(dlsym(lib, "NvEncodeAPICreateInstance"));
  cudaFree(nullptr);  // a context, as an application has before it opens a session

  Encoder e;
  e.api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  if (!create || create(&e.api) != NV_ENC_SUCCESS) {
    std::printf("FAIL NvEncodeAPICreateInstance\n");
    return 1;
  }
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open{};
  open.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  open.apiVersion = NVENCAPI_VERSION;
  open.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  CUcontext ctx = nullptr;   // the card refuses a session without the context it encodes on
  cuCtxGetCurrent(&ctx);
  open.device = ctx;
  if (e.api.nvEncOpenEncodeSessionEx(&open, &e.session) != NV_ENC_SUCCESS) {
    std::printf("FAIL nvEncOpenEncodeSessionEx\n");
    return 1;
  }

  const uint32_t width = 256, height = 128;
  // Both codecs write real (PCM) streams; nvenc_h264.cpp decodes them.
  for (const GUID& codec : {NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID}) {
    if (&codec != &NV_ENC_CODEC_H264_GUID) {
      // A session of its own for each codec.
      e.api.nvEncDestroyEncoder(e.session);
      e.session = nullptr;
      if (e.api.nvEncOpenEncodeSessionEx(&open, &e.session) != NV_ENC_SUCCESS) {
        std::printf("FAIL nvEncOpenEncodeSessionEx\n");
        return 1;
      }
    }
    NV_ENC_INITIALIZE_PARAMS init{};
    init.version = NV_ENC_INITIALIZE_PARAMS_VER;
    init.encodeGUID = codec;
    init.presetGUID = NV_ENC_PRESET_P4_GUID;   // the card refuses an initialisation without a P1-P7 preset
    init.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;   // ... and without a tuning
    init.encodeWidth = width;
    init.encodeHeight = height;
    init.darWidth = width;
    init.darHeight = height;
    init.frameRateNum = 30;
    init.frameRateDen = 1;
    init.enablePTD = 1;
    if (e.api.nvEncInitializeEncoder(e.session, &init) != NV_ENC_SUCCESS) {
      std::printf("FAIL nvEncInitializeEncoder\n");
      e.api.nvEncDestroyEncoder(e.session);
      return 1;
    }
    check_format(e, NV_ENC_BUFFER_FORMAT_ARGB, "ARGB", width, height, height);
    check_format(e, NV_ENC_BUFFER_FORMAT_NV12, "NV12", width, height, height + height / 2);
  }

  workload_contract(e, open);
  e.api.nvEncDestroyEncoder(e.session);
  dlclose(lib);
  if (failures) {
    std::printf("FAIL (%d checks)\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
