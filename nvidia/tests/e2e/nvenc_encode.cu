// NVENC through the documented API, loaded by the bare soname applications
// dlopen, compiled with nvcc against the public Video Codec SDK header.
//
// The shim is not a codec, so nothing here decodes. What it promises -- and
// what encoder stress and corruption checks rely on -- is checked instead: the
// same frame encodes to the same bytes, and a changed byte changes them. For
// the 4:2:0 frame the changed byte is a chroma sample, and the whole frame,
// luma and chroma, is written into the buffer the encoder hands out.
#include <cstdio>
#include <vector>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <nvEncodeAPI.h>

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
                                  NV_ENC_BUFFER_FORMAT fmt, uint32_t width, uint32_t height) {
  NV_ENC_PIC_PARAMS pic{};
  pic.version = NV_ENC_PIC_PARAMS_VER;
  pic.inputBuffer = in;
  pic.outputBitstream = out;
  pic.bufferFmt = fmt;
  pic.inputWidth = width;
  pic.inputHeight = height;
  pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
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

  const auto first = encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height);
  const auto again = encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height);
  if (first.empty()) fail("FAIL %s: encoding produced no bytes%s", name);
  if (first != again) fail("FAIL %s: the same frame encoded differently%s", name);

  unsigned char last = 0;
  cudaMemcpy(&last, frame + n - 1, 1, cudaMemcpyDeviceToHost);
  last ^= 0x5a;
  cudaMemcpy(frame + n - 1, &last, 1, cudaMemcpyHostToDevice);
  const auto changed = encode(e, input.inputBuffer, output.bitstreamBuffer, fmt, width, height);
  if (changed.empty() || changed == first)
    fail("FAIL %s: changing the frame's last byte did not change the output%s", name);
  done();
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
  if (e.api.nvEncOpenEncodeSessionEx(&open, &e.session) != NV_ENC_SUCCESS) {
    std::printf("FAIL nvEncOpenEncodeSessionEx\n");
    return 1;
  }

  const uint32_t width = 256, height = 128;
  NV_ENC_INITIALIZE_PARAMS init{};
  init.version = NV_ENC_INITIALIZE_PARAMS_VER;
  init.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
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

  e.api.nvEncDestroyEncoder(e.session);
  dlclose(lib);
  if (failures) {
    std::printf("FAIL (%d checks)\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
