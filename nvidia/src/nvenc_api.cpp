// libvgpunvenc -- VirtualGPU's NVENC, presented as libnvidia-encode.so.1.
//
// NVENC is a driver component: applications dlopen the bare soname, so putting
// this library first on LD_LIBRARY_PATH stands in for the card's encoder, the
// way libcuda.so.1 stands in for the driver. Everything an application can ask
// of the API answers as NVIDIA's library answered on an RTX 3060 (NVENC API
// 13.0, driver 595; nvidia/tests/e2e/nvenc_api.cpp and its golden file pin
// each status, count, capability, preset configuration and error string): the
// session, the codec / profile / format / capability / preset queries,
// initialisation and its validation, input buffers, registered CUDA resources,
// the locked bitstream's fields, sequence parameters, reconfiguration, and the
// calls that are NVIDIA's "not supported here" on this part.
//
// What is not NVIDIA's: the encoder. A frame is written as an IDR picture whose
// every coding unit is PCM: macroblocks in H.264 (nvenc_h264.hpp, CAVLC) and
// 16x16 coding tree blocks in HEVC (nvenc_hevc.hpp, with the CABAC encoder HEVC
// requires). Both are conformant, lossless streams any decoder returns the input
// from (ffmpeg's do: nvidia/tests/e2e/nvenc_h264.cpp), but not compression --
// rate control, GOP structure, B-frames, the preset and every quality setting are
// accepted and change nothing, and every picture is an IDR, with one idr_pic_id,
// so encoding a frame twice gives the same bytes (encoder SDC tests compare a
// golden bitstream, as pantheon's media_enc_virus does with a forced IDR and the
// parameter sets on every frame). The input is 8-bit 4:2:0 (NV12, YV12, IYUV) or
// 32-bit RGB (ARGB, ABGR; converted to BT.601 limited-range YCbCr, the matrix the
// card applies); the 10-bit and 4:4:4 formats the card takes are refused.
#include <cuda_runtime.h>
#include <nvEncodeAPI.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "nvenc_h264.hpp"
#include "nvenc_h264_enc.hpp"
#include "nvenc_hevc.hpp"

#include "nvenc_tables.inc"

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

// A structure's version word has the API version in its low 16 bits, which the
// card does not look at (a list built with API 14's version is accepted).
bool version_ok(uint32_t got, uint32_t want) { return (got & 0xFFFF0000u) == (want & 0xFFFF0000u); }

// A value out of an application's structure, as the integer it holds: the API's
// enum fields take values outside their enumerators (the card accepts or refuses
// them by number), and loading such a value as its enum type is undefined.
template <class E>
uint32_t raw(const E& e) {
  static_assert(sizeof(E) == sizeof(uint32_t), "an enum the API passes as a 32-bit word");
  uint32_t v;
  std::memcpy(&v, &e, sizeof v);
  return v;
}
template <class E>
constexpr uint32_t num(E e) { return static_cast<uint32_t>(e); }

enum Codec { kNoCodec = -1, kH264 = 0, kHevc = 1 };

bool same_guid(const GUID& a, const GUID& b) { return !std::memcmp(&a, &b, sizeof a); }
int codec_of(const GUID& g) {
  if (same_guid(g, NV_ENC_CODEC_H264_GUID)) return kH264;
  if (same_guid(g, NV_ENC_CODEC_HEVC_GUID)) return kHevc;
  return kNoCodec;
}
int preset_index(const GUID& g) {
  const GUID* all[] = {&NV_ENC_PRESET_P1_GUID, &NV_ENC_PRESET_P2_GUID, &NV_ENC_PRESET_P3_GUID, &NV_ENC_PRESET_P4_GUID,
                       &NV_ENC_PRESET_P5_GUID, &NV_ENC_PRESET_P6_GUID, &NV_ENC_PRESET_P7_GUID};
  for (int i = 0; i < 7; ++i)
    if (same_guid(g, *all[i])) return i + 1;
  return 0;
}

const NV_ENC_BUFFER_FORMAT kFormatsH264[] = {
    NV_ENC_BUFFER_FORMAT_NV12, NV_ENC_BUFFER_FORMAT_YV12, NV_ENC_BUFFER_FORMAT_IYUV, NV_ENC_BUFFER_FORMAT_YUV444,
    NV_ENC_BUFFER_FORMAT_ARGB, NV_ENC_BUFFER_FORMAT_ABGR, NV_ENC_BUFFER_FORMAT_AYUV, NV_ENC_BUFFER_FORMAT_ARGB10,
    NV_ENC_BUFFER_FORMAT_ABGR10};
const NV_ENC_BUFFER_FORMAT kFormatsHevc[] = {
    NV_ENC_BUFFER_FORMAT_NV12, NV_ENC_BUFFER_FORMAT_YV12, NV_ENC_BUFFER_FORMAT_IYUV, NV_ENC_BUFFER_FORMAT_YUV444,
    NV_ENC_BUFFER_FORMAT_YUV420_10BIT, NV_ENC_BUFFER_FORMAT_YUV444_10BIT, NV_ENC_BUFFER_FORMAT_ARGB,
    NV_ENC_BUFFER_FORMAT_ABGR, NV_ENC_BUFFER_FORMAT_AYUV, NV_ENC_BUFFER_FORMAT_ARGB10, NV_ENC_BUFFER_FORMAT_ABGR10};

// The profiles, in the order the card lists them.
const GUID* const kProfilesH264[] = {&NV_ENC_H264_PROFILE_BASELINE_GUID, &NV_ENC_H264_PROFILE_MAIN_GUID,
                                     &NV_ENC_H264_PROFILE_HIGH_GUID,     &NV_ENC_H264_PROFILE_STEREO_GUID,
                                     &NV_ENC_H264_PROFILE_HIGH_444_GUID, &NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID};
const GUID* const kProfilesHevc[] = {&NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID, &NV_ENC_HEVC_PROFILE_MAIN_GUID,
                                     &NV_ENC_HEVC_PROFILE_MAIN10_GUID, &NV_ENC_HEVC_PROFILE_FREXT_GUID};
const GUID* const kPresetGuids[7] = {&NV_ENC_PRESET_P1_GUID, &NV_ENC_PRESET_P2_GUID, &NV_ENC_PRESET_P3_GUID,
                                 &NV_ENC_PRESET_P4_GUID, &NV_ENC_PRESET_P5_GUID, &NV_ENC_PRESET_P6_GUID,
                                 &NV_ENC_PRESET_P7_GUID};

struct Input {
  void* dev = nullptr;
  uint32_t w = 0, h = 0, pitch = 0;
  uint32_t fmt = 0;   // an NV_ENC_BUFFER_FORMAT, as the number the application gave
  bool locked = false;
};
struct Picture {
  std::vector<uint8_t> data;
  uint64_t ts = 0, dur = 0;
  uint32_t frame_idx = 0;
  int pic_struct = 1;
  uint32_t pic_type = NV_ENC_PIC_TYPE_IDR;   // NV_ENC_PIC_TYPE_*
};
struct Output {
  std::deque<Picture> pending;   // encoded, not yet locked: the card queues them per buffer
  Picture current;               // the one locked
  bool locked = false;
};
struct Registered {
  void* dev = nullptr;
  uint32_t w = 0, h = 0, pitch = 0;
  uint32_t fmt = 0;   // an NV_ENC_BUFFER_FORMAT, as the number the application gave
  bool mapped = false;
};
struct Mapped {
  Registered* reg = nullptr;
};

struct Session {
  std::string last_error = "Success.";
  bool initialized = false;
  int codec = kNoCodec;
  uint32_t width = 0, height = 0;
  NV_ENC_INITIALIZE_PARAMS init{};
  NV_ENC_CONFIG config{};
  std::map<void*, std::unique_ptr<Input>> inputs;
  std::map<void*, std::unique_ptr<Output>> outputs;
  std::map<void*, std::unique_ptr<Registered>> registered;
  std::map<void*, std::unique_ptr<Mapped>> mapped;
  uint64_t frames = 0;
  bool sent_parameter_sets = false;
  // The H.264 encoder (everything but the lossless tuning): the compressing encoder with its reference picture,
  // the size and rate it was built for, the picture counters and the rate control state.
  std::unique_ptr<vgpu_nvenc::H264Encoder> h264, h264_trial;
  int h264_w = 0, h264_h = 0, h264_fps_num = 0, h264_fps_den = 0, h264_profile = 0;
  uint64_t since_idr = 0;       // pictures since (and including) the last IDR; 0 before the first picture
  bool force_idr_next = false;  // a reconfiguration asked for it
  double rc_qp = 28;            // the QP the next P picture is coded with
  double rc_bits = 0, rc_target = 0;   // bits spent and bits allowed since the last IDR picture
};

std::mutex g_mu;
std::set<Session*> g_sessions;

Session* find(void* enc) {
  auto* s = static_cast<Session*>(enc);
  return g_sessions.count(s) ? s : nullptr;
}

NVENCSTATUS fail(Session* s, NVENCSTATUS st, const char* message) {
  if (s) s->last_error = message;
  return st;
}

uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

// The pitch (bytes per row) the driver gives a locked input buffer: the row
// rounded up to 64 bytes for the semi-planar and 4:4:4 formats and to 512 for
// the three-plane 4:2:0 ones and the packed 32-bit ones (measured at twelve widths from 145 to 4096
// pixels; the table is in nvidia/tests/e2e/nvenc_api.cpp and fits exactly).
uint32_t buffer_pitch(uint32_t fmt, uint32_t width) {
  switch (fmt) {
    case num(NV_ENC_BUFFER_FORMAT_NV12):
    case num(NV_ENC_BUFFER_FORMAT_YUV444): return align_up(width, 64);
    case num(NV_ENC_BUFFER_FORMAT_YV12):
    case num(NV_ENC_BUFFER_FORMAT_IYUV): return align_up(width, 512);
    default: return align_up(width * 4, 512);
  }
}

// Rows of `pitch` bytes a buffer of this format holds.
uint32_t buffer_rows(uint32_t fmt, uint32_t height) {
  switch (fmt) {
    case num(NV_ENC_BUFFER_FORMAT_NV12):
    case num(NV_ENC_BUFFER_FORMAT_YV12):
    case num(NV_ENC_BUFFER_FORMAT_IYUV):
    case num(NV_ENC_BUFFER_FORMAT_YUV420_10BIT): return height + (height + 1) / 2;
    case num(NV_ENC_BUFFER_FORMAT_YUV444):
    case num(NV_ENC_BUFFER_FORMAT_YUV444_10BIT): return 3 * height;
    default: return height;
  }
}

bool format_listed(int codec, uint32_t fmt) {
  const NV_ENC_BUFFER_FORMAT* l = codec == kHevc ? kFormatsHevc : kFormatsH264;
  const size_t n = codec == kHevc ? sizeof kFormatsHevc / sizeof *kFormatsHevc : sizeof kFormatsH264 / sizeof *kFormatsH264;
  for (size_t i = 0; i < n; ++i)
    if (l[i] == fmt) return true;
  return false;
}

// Whether this encoder can take the format as encode input (the 8-bit 4:2:0 ones
// and 32-bit RGB).
bool encodable(uint32_t fmt) {
  switch (fmt) {
    case num(NV_ENC_BUFFER_FORMAT_NV12):
    case num(NV_ENC_BUFFER_FORMAT_YV12):
    case num(NV_ENC_BUFFER_FORMAT_IYUV):
    case num(NV_ENC_BUFFER_FORMAT_ARGB):
    case num(NV_ENC_BUFFER_FORMAT_ABGR): return true;
    default: return false;
  }
}

// Preset configurations: the card's, as the non-zero words of the structure.
void preset_config(int codec, int preset, int tuning, NV_ENC_PRESET_CONFIG* out) {
  const uint32_t v0 = out->version, v2 = out->presetCfg.version;
  std::memset(out, 0, sizeof *out);
  for (const PresetWords& p : kPresets)
    if (p.codec == codec && p.preset == preset && p.tuning == tuning) {
      uint32_t* words = reinterpret_cast<uint32_t*>(out);
      for (int i = 0; i < p.n; ++i) words[p.w[i][0]] = p.w[i][1];
      break;
    }
  out->version = v0;
  out->presetCfg.version = v2;
}

}  // namespace

/* ---- the function table -------------------------------------------------- */

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncodeAPIGetMaxSupportedVersion(uint32_t* version) {
  if (!version) return NV_ENC_ERR_INVALID_PTR;
  *version = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncOpenEncodeSessionEx(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS* params, void** encoder);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeGUIDCount(void*, uint32_t*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeGUIDs(void*, GUID*, uint32_t, uint32_t*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeProfileGUIDCount(void*, GUID, uint32_t*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeProfileGUIDs(void*, GUID, GUID*, uint32_t, uint32_t*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetInputFormatCount(void*, GUID, uint32_t*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetInputFormats(void*, GUID, NV_ENC_BUFFER_FORMAT*, uint32_t, uint32_t*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeCaps(void*, GUID, NV_ENC_CAPS_PARAM*, int*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodePresetCount(void*, GUID, uint32_t*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodePresetGUIDs(void*, GUID, GUID*, uint32_t, uint32_t*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodePresetConfig(void*, GUID, GUID, NV_ENC_PRESET_CONFIG*);
static NVENCSTATUS NVENCAPI PresetConfigExImpl(void*, GUID, GUID, uint32_t, NV_ENC_PRESET_CONFIG*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncInitializeEncoder(void*, NV_ENC_INITIALIZE_PARAMS*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncCreateInputBuffer(void*, NV_ENC_CREATE_INPUT_BUFFER*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyInputBuffer(void*, NV_ENC_INPUT_PTR);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncCreateBitstreamBuffer(void*, NV_ENC_CREATE_BITSTREAM_BUFFER*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyBitstreamBuffer(void*, NV_ENC_OUTPUT_PTR);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncEncodePicture(void*, NV_ENC_PIC_PARAMS*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncLockBitstream(void*, NV_ENC_LOCK_BITSTREAM*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnlockBitstream(void*, NV_ENC_OUTPUT_PTR);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncLockInputBuffer(void*, NV_ENC_LOCK_INPUT_BUFFER*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnlockInputBuffer(void*, NV_ENC_INPUT_PTR);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeStats(void*, NV_ENC_STAT*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetSequenceParams(void*, NV_ENC_SEQUENCE_PARAM_PAYLOAD*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyEncoder(void*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncRegisterResource(void*, NV_ENC_REGISTER_RESOURCE*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnregisterResource(void*, NV_ENC_REGISTERED_PTR);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncMapInputResource(void*, NV_ENC_MAP_INPUT_RESOURCE*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnmapInputResource(void*, NV_ENC_INPUT_PTR);

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncOpenEncodeSession(void*, uint32_t, void**);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncRegisterAsyncEvent(void*, NV_ENC_EVENT_PARAMS*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnregisterAsyncEvent(void*, NV_ENC_EVENT_PARAMS*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncInvalidateRefFrames(void*, uint64_t);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncReconfigureEncoder(void*, NV_ENC_RECONFIGURE_PARAMS*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncCreateMVBuffer(void*, NV_ENC_CREATE_MV_BUFFER*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyMVBuffer(void*, NV_ENC_OUTPUT_PTR);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncRunMotionEstimationOnly(void*, NV_ENC_MEONLY_PARAMS*);
extern "C" const char* NVENCAPI NvEncGetLastErrorString(void*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncSetIOCudaStreams(void*, NV_ENC_CUSTREAM_PTR, NV_ENC_CUSTREAM_PTR);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetSequenceParamEx(void*, NV_ENC_INITIALIZE_PARAMS*, NV_ENC_SEQUENCE_PARAM_PAYLOAD*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncRestoreEncoderState(void*, NV_ENC_RESTORE_ENCODER_STATE_PARAMS*);
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncLookaheadPicture(void*, NV_ENC_LOOKAHEAD_PIC_PARAMS*);

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncodeAPICreateInstance(NV_ENCODE_API_FUNCTION_LIST* list) {
  if (!list) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(list->version, NV_ENCODE_API_FUNCTION_LIST_VER)) return NV_ENC_ERR_INVALID_VERSION;
  const uint32_t version = list->version;
  std::memset(list, 0, sizeof *list);
  list->version = version;
  list->nvEncOpenEncodeSession = NvEncOpenEncodeSession;
  list->nvEncGetEncodeGUIDCount = NvEncGetEncodeGUIDCount;
  list->nvEncGetEncodeProfileGUIDCount = NvEncGetEncodeProfileGUIDCount;
  list->nvEncGetEncodeProfileGUIDs = NvEncGetEncodeProfileGUIDs;
  list->nvEncGetEncodeGUIDs = NvEncGetEncodeGUIDs;
  list->nvEncGetInputFormatCount = NvEncGetInputFormatCount;
  list->nvEncGetInputFormats = NvEncGetInputFormats;
  list->nvEncGetEncodeCaps = NvEncGetEncodeCaps;
  list->nvEncGetEncodePresetCount = NvEncGetEncodePresetCount;
  list->nvEncGetEncodePresetGUIDs = NvEncGetEncodePresetGUIDs;
  list->nvEncGetEncodePresetConfig = NvEncGetEncodePresetConfig;
  list->nvEncInitializeEncoder = NvEncInitializeEncoder;
  list->nvEncCreateInputBuffer = NvEncCreateInputBuffer;
  list->nvEncDestroyInputBuffer = NvEncDestroyInputBuffer;
  list->nvEncCreateBitstreamBuffer = NvEncCreateBitstreamBuffer;
  list->nvEncDestroyBitstreamBuffer = NvEncDestroyBitstreamBuffer;
  list->nvEncEncodePicture = NvEncEncodePicture;
  list->nvEncLockBitstream = NvEncLockBitstream;
  list->nvEncUnlockBitstream = NvEncUnlockBitstream;
  list->nvEncLockInputBuffer = NvEncLockInputBuffer;
  list->nvEncUnlockInputBuffer = NvEncUnlockInputBuffer;
  list->nvEncGetEncodeStats = NvEncGetEncodeStats;
  list->nvEncGetSequenceParams = NvEncGetSequenceParams;
  list->nvEncMapInputResource = NvEncMapInputResource;
  list->nvEncUnmapInputResource = NvEncUnmapInputResource;
  list->nvEncDestroyEncoder = NvEncDestroyEncoder;
  list->nvEncOpenEncodeSessionEx = NvEncOpenEncodeSessionEx;
  list->nvEncRegisterResource = NvEncRegisterResource;
  list->nvEncUnregisterResource = NvEncUnregisterResource;
  list->nvEncGetEncodePresetConfigEx = reinterpret_cast<PNVENCGETENCODEPRESETCONFIGEX>(PresetConfigExImpl);
  list->nvEncRegisterAsyncEvent = NvEncRegisterAsyncEvent;
  list->nvEncUnregisterAsyncEvent = NvEncUnregisterAsyncEvent;
  list->nvEncInvalidateRefFrames = NvEncInvalidateRefFrames;
  list->nvEncReconfigureEncoder = NvEncReconfigureEncoder;
  list->nvEncCreateMVBuffer = NvEncCreateMVBuffer;
  list->nvEncDestroyMVBuffer = NvEncDestroyMVBuffer;
  list->nvEncRunMotionEstimationOnly = NvEncRunMotionEstimationOnly;
  list->nvEncGetLastErrorString = NvEncGetLastErrorString;
  list->nvEncSetIOCudaStreams = NvEncSetIOCudaStreams;
  list->nvEncGetSequenceParamEx = NvEncGetSequenceParamEx;
  list->nvEncRestoreEncoderState = NvEncRestoreEncoderState;
  list->nvEncLookaheadPicture = NvEncLookaheadPicture;
  return NV_ENC_SUCCESS;
}

/* ---- sessions ---------------------------------------------------------- */

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncOpenEncodeSessionEx(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS* params, void** encoder) {
  if (!params || !encoder) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER)) return NV_ENC_ERR_INVALID_VERSION;
  switch (raw(params->deviceType)) {
    case num(NV_ENC_DEVICE_TYPE_CUDA): break;
    case num(NV_ENC_DEVICE_TYPE_OPENGL): return NV_ENC_ERR_INVALID_DEVICE;
    default: return NV_ENC_ERR_UNSUPPORTED_DEVICE;   // DirectX, and values outside the enum
  }
  if (!params->device) return NV_ENC_ERR_INVALID_PTR;
  std::lock_guard<std::mutex> lock(g_mu);
  auto* s = new Session();
  g_sessions.insert(s);
  *encoder = s;
  if (!quiet())
    std::fprintf(stderr, "[vgpu] virtual NVENC session opened (H.264 I_PCM stream: lossless, uncompressed; see nvidia/docs/libraries.md)\n");
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyEncoder(void* encoder) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = find(encoder);
  if (!s) return NV_ENC_ERR_INVALID_ENCODERDEVICE;
  for (auto& in : s->inputs) cudaFree(in.second->dev);
  g_sessions.erase(s);
  delete s;
  return NV_ENC_SUCCESS;
}

// The text of the last failure a session reported (NVIDIA's, for the calls
// measured; the string persists until another call fails).
extern "C" __attribute__((visibility("default"))) const char* NVENCAPI NvEncGetLastErrorString(void* encoder) {
  std::lock_guard<std::mutex> lock(g_mu);
  Session* s = find(encoder);
  return s ? s->last_error.c_str() : nullptr;
}

/* ---- queries ------------------------------------------------------------ */

#define NEED_SESSION(var)                                                       \
  std::lock_guard<std::mutex> lock(g_mu);                                      \
  Session* var = find(encoder);                                                \
  if (!var) return NV_ENC_ERR_INVALID_ENCODERDEVICE

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeGUIDCount(void* encoder, uint32_t* count) {
  NEED_SESSION(s);
  (void)s;
  if (!count) return NV_ENC_ERR_INVALID_PTR;
  *count = 2;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeGUIDs(void* encoder, GUID* guids, uint32_t size, uint32_t* count) {
  NEED_SESSION(s);
  (void)s;
  if (!guids || !count) return NV_ENC_ERR_INVALID_PTR;
  const GUID* all[] = {&NV_ENC_CODEC_H264_GUID, &NV_ENC_CODEC_HEVC_GUID};
  *count = std::min<uint32_t>(size, 2);
  for (uint32_t i = 0; i < *count; ++i) guids[i] = *all[i];
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeProfileGUIDCount(void* encoder, GUID codec, uint32_t* count) {
  NEED_SESSION(s);
  (void)s;
  if (!count) return NV_ENC_ERR_INVALID_PTR;
  const int c = codec_of(codec);
  if (c == kNoCodec) return NV_ENC_ERR_INVALID_PARAM;
  *count = c == kH264 ? 6 : 4;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeProfileGUIDs(void* encoder, GUID codec, GUID* guids, uint32_t size, uint32_t* count) {
  NEED_SESSION(s);
  (void)s;
  if (!guids || !count) return NV_ENC_ERR_INVALID_PTR;
  const int c = codec_of(codec);
  if (c == kNoCodec) return NV_ENC_ERR_INVALID_PARAM;
  const GUID* const* list = c == kH264 ? kProfilesH264 : kProfilesHevc;
  const uint32_t n = c == kH264 ? 6 : 4;
  *count = std::min(size, n);
  for (uint32_t i = 0; i < *count; ++i) guids[i] = *list[i];
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetInputFormatCount(void* encoder, GUID codec, uint32_t* count) {
  NEED_SESSION(s);
  (void)s;
  if (!count) return NV_ENC_ERR_INVALID_PTR;
  const int c = codec_of(codec);
  if (c == kNoCodec) return NV_ENC_ERR_INVALID_PARAM;
  *count = c == kHevc ? sizeof kFormatsHevc / sizeof *kFormatsHevc : sizeof kFormatsH264 / sizeof *kFormatsH264;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetInputFormats(void* encoder, GUID codec, NV_ENC_BUFFER_FORMAT* formats, uint32_t size, uint32_t* count) {
  NEED_SESSION(s);
  (void)s;
  if (!formats || !count) return NV_ENC_ERR_INVALID_PTR;
  const int c = codec_of(codec);
  if (c == kNoCodec) return NV_ENC_ERR_INVALID_PARAM;
  const NV_ENC_BUFFER_FORMAT* list = c == kHevc ? kFormatsHevc : kFormatsH264;
  const uint32_t n = c == kHevc ? sizeof kFormatsHevc / sizeof *kFormatsHevc : sizeof kFormatsH264 / sizeof *kFormatsH264;
  *count = std::min(size, n);
  for (uint32_t i = 0; i < *count; ++i) formats[i] = list[i];
  return NV_ENC_SUCCESS;
}

// Capabilities 0..60 are the card's; a query past them is NV_ENC_ERR_INVALID_PARAM.
// The structure's version is not checked (a query with version 0 is answered).
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeCaps(void* encoder, GUID codec, NV_ENC_CAPS_PARAM* param, int* value) {
  NEED_SESSION(s);
  (void)s;
  if (!param || !value) return NV_ENC_ERR_INVALID_PTR;
  const int c = codec_of(codec);
  if (c == kNoCodec) return NV_ENC_ERR_INVALID_PARAM;
  const int cap = static_cast<int>(raw(param->capsToQuery));
  if (cap < 0 || cap > 60) return NV_ENC_ERR_INVALID_PARAM;
  *value = (c == kH264 ? kCapsH264 : kCapsHevc)[cap];
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodePresetCount(void* encoder, GUID codec, uint32_t* count) {
  NEED_SESSION(s);
  (void)s;
  if (!count) return NV_ENC_ERR_INVALID_PTR;
  if (codec_of(codec) == kNoCodec) return NV_ENC_ERR_INVALID_PARAM;
  *count = 7;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodePresetGUIDs(void* encoder, GUID codec, GUID* guids, uint32_t size, uint32_t* count) {
  NEED_SESSION(s);
  (void)s;
  if (!guids || !count) return NV_ENC_ERR_INVALID_PTR;
  if (codec_of(codec) == kNoCodec) return NV_ENC_ERR_INVALID_PARAM;
  *count = std::min<uint32_t>(size, 7);
  for (uint32_t i = 0; i < *count; ++i) guids[i] = *kPresetGuids[i];
  return NV_ENC_SUCCESS;
}

// The legacy preset query (the pre-P1-P7 presets) is NVIDIA's NV_ENC_ERR_UNSUPPORTED_PARAM
// for every codec.
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodePresetConfig(void* encoder, GUID, GUID, NV_ENC_PRESET_CONFIG* config) {
  NEED_SESSION(s);
  (void)s;
  if (!config) return NV_ENC_ERR_INVALID_PTR;
  return NV_ENC_ERR_UNSUPPORTED_PARAM;
}
// (The tuning is taken as a number: an application may pass one the enum has no name for.)
static NVENCSTATUS NVENCAPI PresetConfigExImpl(void* encoder, GUID codec, GUID preset, uint32_t tuning_word,
                                                              NV_ENC_PRESET_CONFIG* config) {
  NEED_SESSION(s);
  (void)s;
  if (!config) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(config->version, NV_ENC_PRESET_CONFIG_VER)) return NV_ENC_ERR_INVALID_VERSION;
  const int c = codec_of(codec);
  if (c == kNoCodec) return NV_ENC_ERR_INVALID_PARAM;
  const int p = preset_index(preset);
  const int t = static_cast<int>(tuning_word);
  if (p == 0 || t < 1 || t > 4) return NV_ENC_ERR_UNSUPPORTED_PARAM;
  preset_config(c, p, t, config);
  return NV_ENC_SUCCESS;
}

/* ---- initialisation ------------------------------------------------------- */

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncInitializeEncoder(void* encoder, NV_ENC_INITIALIZE_PARAMS* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_INITIALIZE_PARAMS_VER)) return NV_ENC_ERR_INVALID_VERSION;
  if (params->encodeConfig && !version_ok(params->encodeConfig->version, NV_ENC_CONFIG_VER)) return NV_ENC_ERR_INVALID_VERSION;
  const int codec = codec_of(params->encodeGUID);
  if (same_guid(params->encodeGUID, NV_ENC_CODEC_AV1_GUID)) return fail(s, NV_ENC_ERR_INVALID_PARAM, "EncodeAPI Internal Error.");
  if (codec == kNoCodec) return NV_ENC_ERR_UNSUPPORTED_PARAM;
  const uint32_t w = params->encodeWidth, h = params->encodeHeight;
  if (codec == kH264) {
    if (w < 145 || h < 49) return fail(s, NV_ENC_ERR_INVALID_PARAM, "Frame Dimension less than the minimum supported value.");
    if (w > 4096 || h > 4096) return fail(s, NV_ENC_ERR_INVALID_PARAM, "Frame Dimension greater than the maximum supported value.");
  } else {
    if (w < 129 || h < 33) return fail(s, NV_ENC_ERR_INVALID_PARAM, "Frame dimensions are less than the minimum supported value.");
    if (w > 8192) return fail(s, NV_ENC_ERR_INVALID_PARAM, "Width greater than supported value.");
    if (h > 8192) return fail(s, NV_ENC_ERR_INVALID_PARAM, "Height greater than supported value.");
  }
  if (preset_index(params->presetGUID) == 0)
    return fail(s, NV_ENC_ERR_INVALID_PARAM, "NV_ENC_RC_PARAMS::lowDelayKeyFrameScale is supported with P1-P7 presets\n");
  const int tuning = static_cast<int>(raw(params->tuningInfo));
  if (tuning == 0)
    return fail(s, NV_ENC_ERR_INVALID_PARAM, "Presets P1-P7 are only supported with valid NV_ENC_INITIALIZE_PARAMS::tuningInfo\n");
  if (tuning < 0 || tuning > 4) return fail(s, NV_ENC_ERR_INVALID_PARAM, "Invalid Tuning Info");
  if (tuning == NV_ENC_TUNING_INFO_LOSSLESS && params->encodeConfig &&
      raw(params->encodeConfig->rcParams.rateControlMode) != num(NV_ENC_PARAMS_RC_CONSTQP))
    return fail(s, NV_ENC_ERR_INVALID_PARAM,
                "With lossless preset, RC Mode / Profile not supported with qpPrimeYZeroTransformBypassFlag.");
  // Measured over gop lengths 0..30 and 2^32-1 and frameIntervalP 0..9, both codecs: a
  // non-zero gop length shorter than the frame interval is refused.
  if (params->encodeConfig && params->encodeConfig->gopLength != 0 &&
      params->encodeConfig->frameIntervalP > 0 &&
      static_cast<uint32_t>(params->encodeConfig->frameIntervalP) > params->encodeConfig->gopLength)
    return fail(s, NV_ENC_ERR_INVALID_PARAM, "Gop Length should be greater than number of B frames + 1");
  if ((params->maxEncodeWidth && w > params->maxEncodeWidth) || (params->maxEncodeHeight && h > params->maxEncodeHeight))
    return fail(s, NV_ENC_ERR_INVALID_PARAM, "Encode Width / Height is greater than MaxWidth / MaxHeight.");
  if (params->enableEncodeAsync) return fail(s, NV_ENC_ERR_INVALID_PARAM, "Async mode not supported.");
  if (s->initialized) {   // a second initialisation is accepted and leaves the first in force
    return NV_ENC_SUCCESS;
  }
  s->codec = codec;
  s->width = w;
  s->height = h;
  s->init = *params;
  s->init.encodeConfig = nullptr;
  if (params->encodeConfig) {
    s->config = *params->encodeConfig;
  } else {
    NV_ENC_PRESET_CONFIG pc{};
    pc.version = NV_ENC_PRESET_CONFIG_VER;
    pc.presetCfg.version = NV_ENC_CONFIG_VER;
    preset_config(codec, preset_index(params->presetGUID), tuning, &pc);
    s->config = pc.presetCfg;
  }
  s->initialized = true;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncReconfigureEncoder(void* encoder, NV_ENC_RECONFIGURE_PARAMS* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_RECONFIGURE_PARAMS_VER)) return NV_ENC_ERR_INVALID_VERSION;
  if (!s->initialized) return NV_ENC_ERR_DEVICE_NOT_EXIST;
  // No validation: the card accepts any new size (measured: larger than the
  // initial one, with no maximum set, and smaller).
  if (params->reInitEncodeParams.encodeWidth) s->width = params->reInitEncodeParams.encodeWidth;
  if (params->reInitEncodeParams.encodeHeight) s->height = params->reInitEncodeParams.encodeHeight;
  // The rate control and GOP settings of the new configuration take effect, and a reset (or a forced IDR) starts a new
  // coded sequence; a new size or frame rate does too, in the encoder.
  if (params->reInitEncodeParams.encodeConfig) s->config = *params->reInitEncodeParams.encodeConfig;
  if (params->reInitEncodeParams.frameRateNum) s->init.frameRateNum = params->reInitEncodeParams.frameRateNum;
  if (params->reInitEncodeParams.frameRateDen) s->init.frameRateDen = params->reInitEncodeParams.frameRateDen;
  if (params->resetEncoder || params->forceIDR) s->force_idr_next = true;
  return NV_ENC_SUCCESS;
}

/* ---- buffers --------------------------------------------------------------- */

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncCreateInputBuffer(void* encoder, NV_ENC_CREATE_INPUT_BUFFER* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_CREATE_INPUT_BUFFER_VER)) return NV_ENC_ERR_INVALID_VERSION;
  if (!s->initialized) return NV_ENC_ERR_DEVICE_NOT_EXIST;
  if (!params->width || !params->height) return NV_ENC_ERR_OUT_OF_MEMORY;
  const uint32_t want_fmt = raw(params->bufferFmt);
  if (!format_listed(s->codec, want_fmt) ||
      (s->codec == kH264 && (want_fmt == num(NV_ENC_BUFFER_FORMAT_YUV420_10BIT) || want_fmt == num(NV_ENC_BUFFER_FORMAT_YUV444_10BIT))))
    return NV_ENC_ERR_INVALID_PARAM;
  auto in = std::make_unique<Input>();
  in->w = params->width;
  in->h = params->height;
  in->fmt = want_fmt;
  in->pitch = buffer_pitch(in->fmt, in->w);
  const size_t bytes = static_cast<size_t>(in->pitch) * buffer_rows(in->fmt, in->h);
  // Managed memory: the CPU writes a locked buffer directly, as on the card, and
  // a kernel may write the same pointer.
  if (cudaMallocManaged(&in->dev, bytes) != cudaSuccess || !in->dev) return NV_ENC_ERR_OUT_OF_MEMORY;
  std::memset(in->dev, 0, bytes);
  void* handle = in->dev;   // the handle is the device pointer itself
  s->inputs[handle] = std::move(in);
  params->inputBuffer = handle;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyInputBuffer(void* encoder, NV_ENC_INPUT_PTR input) {
  NEED_SESSION(s);
  if (!input) return NV_ENC_ERR_INVALID_PARAM;
  auto it = s->inputs.find(input);
  if (it == s->inputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  cudaFree(it->second->dev);
  s->inputs.erase(it);
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncLockInputBuffer(void* encoder, NV_ENC_LOCK_INPUT_BUFFER* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->inputs.find(params->inputBuffer);
  if (!params->inputBuffer || it == s->inputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  it->second->locked = true;
  params->bufferDataPtr = it->second->dev;   // managed memory: a kernel may write it
  params->pitch = it->second->pitch;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnlockInputBuffer(void* encoder, NV_ENC_INPUT_PTR input) {
  NEED_SESSION(s);
  auto it = s->inputs.find(input);
  if (!input || it == s->inputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  it->second->locked = false;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncCreateBitstreamBuffer(void* encoder, NV_ENC_CREATE_BITSTREAM_BUFFER* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_CREATE_BITSTREAM_BUFFER_VER)) return NV_ENC_ERR_INVALID_VERSION;
  if (!s->initialized) return NV_ENC_ERR_DEVICE_NOT_EXIST;
  auto out = std::make_unique<Output>();
  void* handle = out.get();
  s->outputs[handle] = std::move(out);
  params->bitstreamBuffer = handle;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyBitstreamBuffer(void* encoder, NV_ENC_OUTPUT_PTR bitstream) {
  NEED_SESSION(s);
  if (!bitstream) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->outputs.find(bitstream);
  if (it == s->outputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  s->outputs.erase(it);
  return NV_ENC_SUCCESS;
}

/* ---- registered resources --------------------------------------------------- */

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncRegisterResource(void* encoder, NV_ENC_REGISTER_RESOURCE* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_REGISTER_RESOURCE_VER)) return NV_ENC_ERR_INVALID_VERSION;
  // The card registers whatever it is given -- a null pointer, an undefined
  // format, a zero width -- and fails later, if at all.
  auto r = std::make_unique<Registered>();
  r->dev = params->resourceToRegister;
  r->w = params->width;
  r->h = params->height;
  r->pitch = params->pitch;
  r->fmt = raw(params->bufferFormat);
  void* handle = r.get();
  s->registered[handle] = std::move(r);
  params->registeredResource = handle;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnregisterResource(void* encoder, NV_ENC_REGISTERED_PTR resource) {
  NEED_SESSION(s);
  if (!resource) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->registered.find(resource);
  if (it == s->registered.end()) return NV_ENC_ERR_RESOURCE_NOT_REGISTERED;
  for (auto m = s->mapped.begin(); m != s->mapped.end();)
    m = m->second->reg == it->second.get() ? s->mapped.erase(m) : std::next(m);
  s->registered.erase(it);
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncMapInputResource(void* encoder, NV_ENC_MAP_INPUT_RESOURCE* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_MAP_INPUT_RESOURCE_VER)) return NV_ENC_ERR_INVALID_VERSION;
  auto it = s->registered.find(params->registeredResource);
  if (!params->registeredResource || it == s->registered.end()) return NV_ENC_ERR_RESOURCE_NOT_REGISTERED;
  auto m = std::make_unique<Mapped>();
  m->reg = it->second.get();
  void* handle = m.get();
  s->mapped[handle] = std::move(m);
  params->mappedResource = handle;
  std::memcpy(&params->mappedBufferFmt, &it->second->fmt, sizeof it->second->fmt);
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnmapInputResource(void* encoder, NV_ENC_INPUT_PTR mapped) {
  NEED_SESSION(s);
  auto it = s->mapped.find(mapped);
  if (!mapped || it == s->mapped.end()) return NV_ENC_ERR_RESOURCE_NOT_MAPPED;
  s->mapped.erase(it);
  return NV_ENC_SUCCESS;
}

/* ---- encoding ----------------------------------------------------------------- */

namespace {

// 8-bit planar 4:2:0 as the encoder reads it: sample accessors over device memory
// copied to the host.
struct Frame {
  std::vector<uint8_t> y, u, v;
  int w = 0, h = 0, cw = 0, ch = 0;
};

// BT.601 limited range, the matrix the card's encoder applies to ARGB / ABGR input
// (flat colours encoded at QP 1 and decoded: red 81/90/240, green 145/54/34, blue
// 41/240/110, white 235/128/128, grey 128 gives 126/128/128): Y = 16 + (65.481 R +
// 128.553 G + 24.966 B) / 255 and the chroma likewise, rounded to nearest.
uint8_t luma_of(int r, int g, int b) {
  return static_cast<uint8_t>(std::lround(16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0));
}
uint8_t chroma_of(double sum, int n) {
  return static_cast<uint8_t>(std::min(255L, std::max(0L, std::lround(128.0 + sum / (255.0 * n)))));
}

Frame read_frame(const uint8_t* dev, uint32_t pitch, uint32_t fmt, int w, int h, const uint32_t chroma_rows_offset) {
  Frame f;
  f.w = w;
  f.h = h;
  f.cw = (w + 1) / 2;
  f.ch = (h + 1) / 2;
  f.y.resize(static_cast<size_t>(w) * h);
  f.u.resize(static_cast<size_t>(f.cw) * f.ch);
  f.v.resize(f.u.size());
  auto copy2d = [&](std::vector<uint8_t>& dst, const uint8_t* src, size_t src_pitch, int width, int rows) {
    cudaMemcpy2D(dst.data(), width, src, src_pitch, width, rows, cudaMemcpyDefault);
  };
  switch (fmt) {
    case num(NV_ENC_BUFFER_FORMAT_NV12): {
      copy2d(f.y, dev, pitch, w, h);
      std::vector<uint8_t> uv(static_cast<size_t>(f.cw) * 2 * f.ch);
      copy2d(uv, dev + static_cast<size_t>(pitch) * chroma_rows_offset, pitch, f.cw * 2, f.ch);
      for (size_t i = 0; i < f.u.size(); ++i) {
        f.u[i] = uv[2 * i];
        f.v[i] = uv[2 * i + 1];
      }
      break;
    }
    case num(NV_ENC_BUFFER_FORMAT_YV12):
    case num(NV_ENC_BUFFER_FORMAT_IYUV): {
      copy2d(f.y, dev, pitch, w, h);
      const uint8_t* p1 = dev + static_cast<size_t>(pitch) * chroma_rows_offset;
      const uint8_t* p2 = p1 + static_cast<size_t>(pitch / 2) * f.ch;
      std::vector<uint8_t>& first = fmt == num(NV_ENC_BUFFER_FORMAT_IYUV) ? f.u : f.v;
      std::vector<uint8_t>& second = fmt == num(NV_ENC_BUFFER_FORMAT_IYUV) ? f.v : f.u;
      copy2d(first, p1, pitch / 2, f.cw, f.ch);
      copy2d(second, p2, pitch / 2, f.cw, f.ch);
      break;
    }
    default: {   // ARGB / ABGR: 32-bit words, converted
      std::vector<uint8_t> px(static_cast<size_t>(w) * h * 4);
      copy2d(px, dev, pitch, w * 4, h);
      const bool abgr = fmt == num(NV_ENC_BUFFER_FORMAT_ABGR);
      auto rgb = [&](int x, int y, int* r, int* g, int* b) {
        const uint8_t* p = &px[(static_cast<size_t>(y) * w + x) * 4];
        // ARGB is a word with A in the top byte: B, G, R, A in memory; ABGR: R, G, B, A.
        *r = abgr ? p[0] : p[2];
        *g = p[1];
        *b = abgr ? p[2] : p[0];
      };
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          int r, g, b;
          rgb(x, y, &r, &g, &b);
          f.y[static_cast<size_t>(y) * w + x] = luma_of(r, g, b);
        }
      for (int y = 0; y < f.ch; ++y)
        for (int x = 0; x < f.cw; ++x) {
          int sr = 0, sg = 0, sb = 0, n = 0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
              int r, g, b;
              rgb(std::min(2 * x + dx, w - 1), std::min(2 * y + dy, h - 1), &r, &g, &b);
              sr += r; sg += g; sb += b; ++n;
            }
          // Chroma from the block's mean colour.
          f.u[static_cast<size_t>(y) * f.cw + x] = chroma_of(-37.797 * sr - 74.203 * sg + 112.0 * sb, n);
          f.v[static_cast<size_t>(y) * f.cw + x] = chroma_of(112.0 * sr - 93.786 * sg - 18.214 * sb, n);
        }
    }
  }
  return f;
}

std::vector<uint8_t> parameter_sets_of(int codec, int width, int height, uint32_t rate_num, uint32_t rate_den) {
  const int fps_num = rate_num ? static_cast<int>(rate_num) : 30, fps_den = rate_den ? static_cast<int>(rate_den) : 1;
  if (codec == kHevc) {
    vgpu_nvenc::HevcStream st;
    st.width = width;
    st.height = height;
    st.fps_num = fps_num;
    st.fps_den = fps_den;
    return st.parameter_sets();
  }
  vgpu_nvenc::H264Stream st;
  st.width = width;
  st.height = height;
  st.fps_num = fps_num;
  st.fps_den = fps_den;
  return st.parameter_sets();
}

std::vector<uint8_t> parameter_sets(const Session& s) {
  return parameter_sets_of(s.codec, static_cast<int>(s.width), static_cast<int>(s.height), s.init.frameRateNum, s.init.frameRateDen);
}

// ---- the compressing H.264 encoder ----------------------------------------------------------------------------------

// The lossless tuning (and the lossless flag of the H.264 configuration) is the only mode the card encodes without
// loss, and it is the one mode here that stays the PCM stream: transform bypass is a High 4:4:4 Predictive tool.
bool h264_lossless(const Session& s) {
  return raw(s.init.tuningInfo) == num(NV_ENC_TUNING_INFO_LOSSLESS) ||
         s.config.encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag != 0;
}

int h264_profile_idc(const NV_ENC_CONFIG& c) {
  if (same_guid(c.profileGUID, NV_ENC_H264_PROFILE_BASELINE_GUID)) return 66;
  if (same_guid(c.profileGUID, NV_ENC_H264_PROFILE_MAIN_GUID)) return 77;
  return 100;   // High, and what the automatic choice picks
}

vgpu_nvenc::EncPicture to_enc_picture(const Frame& f) {
  vgpu_nvenc::EncPicture p;
  p.w = f.w;
  p.h = f.h;
  p.y = f.y;
  p.u = f.u;
  p.v = f.v;
  return p;
}

// Encodes one picture with the compressing encoder: the picture type from the GOP settings (and the application's flags),
// the QP from the rate control mode. Returns the bytes (parameter sets first when they are due) and the picture type.
std::vector<uint8_t> encode_h264(Session& s, const Frame& f, const NV_ENC_PIC_PARAMS& pp, int rate_num, int rate_den, uint32_t* pic_type) {
  const NV_ENC_CONFIG& c = s.config;
  const int profile = h264_profile_idc(c);
  if (!s.h264 || s.h264_w != f.w || s.h264_h != f.h || s.h264_fps_num != rate_num || s.h264_fps_den != rate_den || s.h264_profile != profile) {
    s.h264 = std::make_unique<vgpu_nvenc::H264Encoder>(f.w, f.h, rate_num, rate_den, profile, true);
    s.h264_trial.reset();
    s.h264_w = f.w;
    s.h264_h = f.h;
    s.h264_fps_num = rate_num;
    s.h264_fps_den = rate_den;
    s.h264_profile = profile;
    s.since_idr = 0;
    s.sent_parameter_sets = false;   // a new sequence: the decoder needs its headers
  }
  const NV_ENC_CONFIG_H264& h = c.encodeCodecConfig.h264Config;
  const uint32_t gop = c.gopLength;
  const uint32_t idr_period = h.idrPeriod ? h.idrPeriod : gop;
  const bool intra_only = c.frameIntervalP == 0;
  const bool periodic = idr_period != 0 && idr_period != NVENC_INFINITE_GOPLENGTH;
  const uint32_t flags = pp.encodePicFlags;
  vgpu_nvenc::PicType t;
  if (s.init.enablePTD) {
    if (s.since_idr == 0 || s.force_idr_next || (flags & NV_ENC_PIC_FLAG_FORCEIDR) || (periodic && s.since_idr >= idr_period))
      t = vgpu_nvenc::PicType::kIdr;
    else if (intra_only || (flags & NV_ENC_PIC_FLAG_FORCEINTRA))
      t = vgpu_nvenc::PicType::kIntra;   // frameIntervalP 0: the card writes an IDR picture and then I pictures (measured)
    else
      t = vgpu_nvenc::PicType::kInter;
  } else {
    // the application decides (NV_ENC_PIC_PARAMS::pictureType); a first picture is always an IDR
    const uint32_t pt = raw(pp.pictureType);
    t = s.since_idr == 0 || pt == num(NV_ENC_PIC_TYPE_IDR) ? vgpu_nvenc::PicType::kIdr
        : (pt == num(NV_ENC_PIC_TYPE_I) ? vgpu_nvenc::PicType::kIntra : vgpu_nvenc::PicType::kInter);
  }
  s.force_idr_next = false;
  const bool intra = t != vgpu_nvenc::PicType::kInter;

  // ---- the QP
  const NV_ENC_RC_PARAMS& rc = c.rcParams;
  const uint32_t mode = raw(rc.rateControlMode);
  const long bitrate = rc.averageBitRate;
  const double frame_bits = bitrate > 0 ? static_cast<double>(bitrate) * rate_den / std::max(rate_num, 1) : 0;
  // an I picture takes about four times a P picture's bits; with a short GOP its share of the budget is smaller
  const double gop_len = periodic ? static_cast<double>(std::max<uint32_t>(idr_period, 1)) : 1e9;
  const double p_bits = frame_bits * gop_len / (gop_len + 3.0);
  const double i_bits = intra_only || (periodic && idr_period <= 1) ? frame_bits : 4.0 * p_bits;
  int qp;
  const vgpu_nvenc::EncPicture pic = to_enc_picture(f);
  if (mode == num(NV_ENC_PARAMS_RC_CONSTQP) || bitrate <= 0) {
    qp = intra ? rc.constQP.qpIntra : rc.constQP.qpInterP;
  } else if (intra) {
    // a stateless search, so that one frame always encodes to the same bytes: trial encodes on a scratch encoder
    if (!s.h264_trial) s.h264_trial = std::make_unique<vgpu_nvenc::H264Encoder>(f.w, f.h, rate_num, rate_den, profile, true);
    qp = vgpu_nvenc::initial_qp_for(f.w, f.h, rate_num, rate_den, static_cast<long>(i_bits * rate_num / std::max(rate_den, 1)));
    for (int trial = 0; trial < 4; ++trial) {
      vgpu_nvenc::EncStats st;
      s.h264_trial->encode(pic, vgpu_nvenc::PicType::kIdr, qp, &st);
      const double ratio = static_cast<double>(st.bytes) * 8 / i_bits;
      if (ratio > 0.85 && ratio < 1.2) break;
      const int next = std::min(51, std::max(8, qp + static_cast<int>(std::lround(6.0 * std::log2(ratio)))));
      if (next == qp) break;
      qp = next;
    }
  } else {
    qp = static_cast<int>(std::lround(s.rc_qp));
  }
  if (rc.enableMinQP) qp = std::max<int>(qp, intra ? rc.minQP.qpIntra : rc.minQP.qpInterP);
  if (rc.enableMaxQP) qp = std::min<int>(qp, intra ? rc.maxQP.qpIntra : rc.maxQP.qpInterP);
  qp = std::min(51, std::max(0, qp));

  vgpu_nvenc::EncStats st;
  std::vector<uint8_t> picture = s.h264->encode(pic, t, qp, &st);
  // ---- rate control state
  if (frame_bits > 0 && mode != num(NV_ENC_PARAMS_RC_CONSTQP)) {
    const double bits = static_cast<double>(picture.size()) * 8;
    if (t == vgpu_nvenc::PicType::kIdr) {
      s.rc_bits = bits;
      s.rc_target = i_bits;
      s.rc_qp = std::min(51, qp + 2);
    } else {
      s.rc_bits += bits;
      s.rc_target += p_bits;
      if (!intra) {
        const double ratio = s.rc_bits / std::max(s.rc_target, 1.0);
        s.rc_qp = std::min(51.0, std::max(8.0, s.rc_qp + std::min(3.0, std::max(-3.0, 3.5 * std::log2(ratio)))));
      }
    }
  }
  s.since_idr = t == vgpu_nvenc::PicType::kIdr ? 1 : s.since_idr + 1;
  *pic_type = t == vgpu_nvenc::PicType::kIdr ? num(NV_ENC_PIC_TYPE_IDR) : (t == vgpu_nvenc::PicType::kIntra ? num(NV_ENC_PIC_TYPE_I) : num(NV_ENC_PIC_TYPE_P));
  std::vector<uint8_t> out;
  size_t skip = 0;
  if (!s.sent_parameter_sets || (flags & NV_ENC_PIC_FLAG_OUTPUT_SPSPPS) || (h.repeatSPSPPS && t == vgpu_nvenc::PicType::kIdr)) {
    out = s.h264->parameter_sets();
    // The digest SEI message of an intra picture is the first NAL unit of its access unit, and needs the four-byte start code,
    // only when no parameter sets come before it (Annex B, zero_byte).
    if (picture.size() > 4 && picture[3] == 1 && (picture[4] & 31) == 6) skip = 1;
  }
  out.insert(out.end(), picture.begin() + static_cast<std::ptrdiff_t>(skip), picture.end());
  s.sent_parameter_sets = true;
  return out;
}

}  // namespace

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetSequenceParams(void* encoder, NV_ENC_SEQUENCE_PARAM_PAYLOAD* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER)) return NV_ENC_ERR_INVALID_VERSION;
  if (!s->initialized) return NV_ENC_ERR_DEVICE_NOT_EXIST;
  if (!params->spsppsBuffer || !params->outSPSPPSPayloadSize) return NV_ENC_ERR_INVALID_PARAM;
  const std::vector<uint8_t> ps = parameter_sets(*s);
  if (params->inBufferSize < ps.size()) return NV_ENC_ERR_OUT_OF_MEMORY;
  std::memcpy(params->spsppsBuffer, ps.data(), ps.size());
  *params->outSPSPPSPayloadSize = static_cast<uint32_t>(ps.size());
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncEncodePicture(void* encoder, NV_ENC_PIC_PARAMS* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_PIC_PARAMS_VER)) return NV_ENC_ERR_INVALID_VERSION;
  if (!s->initialized) return NV_ENC_ERR_DEVICE_NOT_EXIST;
  if (params->encodePicFlags & NV_ENC_PIC_FLAG_EOS) return NV_ENC_SUCCESS;   // nothing is held back
  if (!params->inputBuffer || !params->outputBitstream) return NV_ENC_ERR_INVALID_PARAM;
  const int ps = static_cast<int>(raw(params->pictureStruct));
  if (ps < 1 || ps > 3) return fail(s, NV_ENC_ERR_INVALID_PARAM, "Invalid value for NV_ENC_PIC_STRUCT.");
  auto out = s->outputs.find(params->outputBitstream);
  if (out == s->outputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  const uint8_t* dev;
  uint32_t pitch, in_w, in_h;
  uint32_t fmt;
  if (auto in = s->inputs.find(params->inputBuffer); in != s->inputs.end()) {
    dev = static_cast<const uint8_t*>(in->second->dev);
    pitch = in->second->pitch;
    in_w = in->second->w;
    in_h = in->second->h;
    fmt = in->second->fmt;
  } else if (auto m = s->mapped.find(params->inputBuffer); m != s->mapped.end() && m->second->reg->dev) {
    const Registered* r = m->second->reg;
    dev = static_cast<const uint8_t*>(r->dev);
    pitch = r->pitch;
    in_w = r->w;
    in_h = r->h;
    fmt = r->fmt;
  } else {
    return NV_ENC_ERR_INVALID_PARAM;
  }
  if (!encodable(fmt)) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] NVENC: this input format is not encoded by VirtualGPU (8-bit 4:2:0 and 32-bit RGB only)\n");
    return NV_ENC_ERR_UNSUPPORTED_PARAM;
  }
  // The picture is the size the session was initialised (or reconfigured) to,
  // read from the top-left of the input.
  const int w = static_cast<int>(std::min<uint32_t>(s->width, in_w)), h = static_cast<int>(std::min<uint32_t>(s->height, in_h));
  if (w <= 0 || h <= 0) return NV_ENC_ERR_INVALID_PARAM;
  const Frame f = read_frame(dev, pitch, fmt, w, h, in_h);
  const int rate_num = s->init.frameRateNum ? static_cast<int>(s->init.frameRateNum) : 30;
  const int rate_den = s->init.frameRateDen ? static_cast<int>(s->init.frameRateDen) : 1;
  const auto luma = [&](int x, int y) { return f.y[static_cast<size_t>(y) * f.w + x]; };
  const auto chroma = [&](int plane, int x, int y) { return (plane ? f.v : f.u)[static_cast<size_t>(y) * f.cw + x]; };
  std::vector<uint8_t> bits, picture;
  uint32_t pic_type = num(NV_ENC_PIC_TYPE_IDR);   // the PCM streams: every picture is an IDR
  if (s->codec == kHevc) {
    vgpu_nvenc::HevcStream st;
    st.width = w;
    st.height = h;
    st.fps_num = rate_num;
    st.fps_den = rate_den;
    if (!s->sent_parameter_sets || (params->encodePicFlags & NV_ENC_PIC_FLAG_OUTPUT_SPSPPS) || s->config.encodeCodecConfig.hevcConfig.repeatSPSPPS)
      bits = st.parameter_sets();
    picture = st.idr(luma, chroma);
  } else {
    if (h264_lossless(*s)) {
      vgpu_nvenc::H264Stream st;
      st.width = w;
      st.height = h;
      st.fps_num = rate_num;
      st.fps_den = rate_den;
      if (!s->sent_parameter_sets || (params->encodePicFlags & NV_ENC_PIC_FLAG_OUTPUT_SPSPPS) || s->config.encodeCodecConfig.h264Config.repeatSPSPPS)
        bits = st.parameter_sets();
      picture = st.idr(luma, chroma, 0);
    } else {
      bits = encode_h264(*s, f, *params, rate_num, rate_den, &pic_type);
    }
  }
  s->sent_parameter_sets = true;
  bits.insert(bits.end(), picture.begin(), picture.end());
  Output& o = *out->second;
  o.pending.push_back({std::move(bits), params->inputTimeStamp, params->inputDuration, static_cast<uint32_t>(s->frames++), ps, pic_type});
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncLockBitstream(void* encoder, NV_ENC_LOCK_BITSTREAM* params) {
  NEED_SESSION(s);
  if (!params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_LOCK_BITSTREAM_VER)) return NV_ENC_ERR_INVALID_VERSION;
  auto it = s->outputs.find(params->outputBitstream);
  if (!params->outputBitstream || it == s->outputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  Output& o = *it->second;
  if (o.pending.empty()) return NV_ENC_ERR_INVALID_CALL;
  o.current = std::move(o.pending.front());
  o.pending.pop_front();
  o.locked = true;
  params->bitstreamBufferPtr = o.current.data.data();
  params->bitstreamSizeInBytes = static_cast<uint32_t>(o.current.data.size());
  params->outputTimeStamp = o.current.ts;
  params->outputDuration = o.current.dur;
  params->frameIdx = o.current.frame_idx;
  params->pictureType = static_cast<NV_ENC_PIC_TYPE>(o.current.pic_type);
  params->pictureStruct = static_cast<NV_ENC_PIC_STRUCT>(o.current.pic_struct);
  params->hwEncodeStatus = 2;
  params->numSlices = 1;
  params->temporalId = 0;
  params->sliceOffsets = nullptr;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnlockBitstream(void* encoder, NV_ENC_OUTPUT_PTR bitstream) {
  NEED_SESSION(s);
  if (!bitstream) return NV_ENC_ERR_INVALID_PTR;
  auto it = s->outputs.find(bitstream);
  if (it == s->outputs.end()) return NV_ENC_ERR_INVALID_PARAM;
  it->second->locked = false;
  return NV_ENC_SUCCESS;
}

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetEncodeStats(void* encoder, NV_ENC_STAT* stats) {
  NEED_SESSION(s);
  (void)s;
  if (!stats) return NV_ENC_ERR_INVALID_PTR;
  return NV_ENC_ERR_INVALID_PARAM;   // the card's answer to the statistics query of a bitstream buffer
}

/* ---- the rest of the table ---------------------------------------------------- */

VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncOpenEncodeSession(void* device, uint32_t device_type, void** encoder) {
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS p{};
  p.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  p.device = device;
  p.deviceType = static_cast<NV_ENC_DEVICE_TYPE>(device_type);
  p.apiVersion = NVENCAPI_VERSION;
  return NvEncOpenEncodeSessionEx(&p, encoder);
}

// Measured on the card: these calls take a valid session and nothing else is
// checked; the ones for features this part (or this build) lacks answer so.
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncInvalidateRefFrames(void* encoder, uint64_t) {
  NEED_SESSION(s);
  (void)s;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncRegisterAsyncEvent(void* encoder, NV_ENC_EVENT_PARAMS*) {
  NEED_SESSION(s);
  (void)s;
  return NV_ENC_ERR_UNIMPLEMENTED;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncUnregisterAsyncEvent(void* encoder, NV_ENC_EVENT_PARAMS*) {
  NEED_SESSION(s);
  (void)s;
  return NV_ENC_ERR_UNIMPLEMENTED;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncSetIOCudaStreams(void* encoder, NV_ENC_CUSTREAM_PTR, NV_ENC_CUSTREAM_PTR) {
  NEED_SESSION(s);
  (void)s;
  return NV_ENC_SUCCESS;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncCreateMVBuffer(void* encoder, NV_ENC_CREATE_MV_BUFFER*) {
  NEED_SESSION(s);
  (void)s;
  return NV_ENC_ERR_UNIMPLEMENTED;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncDestroyMVBuffer(void* encoder, NV_ENC_OUTPUT_PTR) {
  NEED_SESSION(s);
  (void)s;
  return NV_ENC_ERR_INVALID_PTR;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncRunMotionEstimationOnly(void* encoder, NV_ENC_MEONLY_PARAMS*) {
  NEED_SESSION(s);
  (void)s;
  return NV_ENC_ERR_INVALID_PTR;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncLookaheadPicture(void* encoder, NV_ENC_LOOKAHEAD_PIC_PARAMS*) {
  NEED_SESSION(s);
  (void)s;
  return NV_ENC_ERR_INVALID_PARAM;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncRestoreEncoderState(void* encoder, NV_ENC_RESTORE_ENCODER_STATE_PARAMS*) {
  NEED_SESSION(s);
  (void)s;
  return NV_ENC_ERR_INVALID_PARAM;
}
VGPU_EXPORT NVENCSTATUS NVENCAPI NvEncGetSequenceParamEx(void* encoder, NV_ENC_INITIALIZE_PARAMS* init, NV_ENC_SEQUENCE_PARAM_PAYLOAD* params) {
  NEED_SESSION(s);
  (void)s;
  if (!init || !params) return NV_ENC_ERR_INVALID_PTR;
  if (!version_ok(params->version, NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER)) return NV_ENC_ERR_INVALID_VERSION;
  if (!params->spsppsBuffer || !params->outSPSPPSPayloadSize) return NV_ENC_ERR_INVALID_PARAM;
  const int codec = codec_of(init->encodeGUID);
  if (codec == kNoCodec) return NV_ENC_ERR_UNSUPPORTED_PARAM;
  if (init->encodeWidth == 0 || init->encodeHeight == 0) return NV_ENC_ERR_INVALID_PARAM;
  const std::vector<uint8_t> ps = parameter_sets_of(codec, static_cast<int>(init->encodeWidth), static_cast<int>(init->encodeHeight),
                                                    init->frameRateNum, init->frameRateDen);
  if (params->inBufferSize < ps.size()) return NV_ENC_ERR_OUT_OF_MEMORY;
  std::memcpy(params->spsppsBuffer, ps.data(), ps.size());
  *params->outSPSPPSPayloadSize = static_cast<uint32_t>(ps.size());
  return NV_ENC_SUCCESS;
}
