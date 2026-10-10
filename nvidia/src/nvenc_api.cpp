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
#include "nvenc_hevc_enc.hpp"
#include "nvenc_rc.hpp"

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
  uint32_t slices = 1;
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

// 8-bit planar 4:2:0 as the encoder reads it: sample accessors over device memory
// copied to the host.
struct Frame {
  std::vector<uint8_t> y, u, v;
  int w = 0, h = 0, cw = 0, ch = 0;
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
  // The compressing encoder (everything but the lossless tuning): its reference pictures, the size, rate and profile it was built for, the
  // picture counters and the rate control state.
  std::unique_ptr<vgpu_nvenc::VideoEncoder> enc, trial;
  int enc_codec = -1, enc_w = 0, enc_h = 0, enc_fps_num = 0, enc_fps_den = 0, enc_profile = 0;
  uint64_t enc_sig = 0;         // the tools the encoder was built with (entropy mode, transform, slices, references, B pictures)
  uint64_t since_idr = 0;       // pictures since (and including) the last IDR picture, in display order; 0 before the first picture
  bool force_idr_next = false;  // a reconfiguration asked for it
  vgpu_nvenc::RateController rc;   // bit rate control (CBR and VBR)
  uint64_t rc_sig = 0;             // the settings the controller was configured from
  // B pictures: the pictures that wait for the P picture that follows them in display order, and the output buffers of the calls that
  // returned NV_ENC_ERR_NEED_MORE_INPUT (the card fills those, in the order of the calls, when the group is coded).
  struct Waiting {
    Frame frame;
    uint64_t ts = 0, dur = 0;
    int pic_struct = 1;
    int disp = 0;               // display index since the last IDR picture
    bool want_ps = false;       // the call asked for the parameter sets with this picture
  };
  std::vector<Waiting> waiting;
  std::deque<void*> outstanding;
  int disp = 0;                 // display index of the next picture
  int last_anchor = 0;          // display index of the last picture coded as I or P
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

// ---- the compressing encoders (H.264 and HEVC) -----------------------------------------------------------------------

// The lossless tuning (and the lossless flag of the H.264 configuration) is the only mode the card encodes without
// loss, and it is the one mode here that stays the PCM stream: transform bypass is a High 4:4:4 Predictive tool.
bool lossless_of(int codec, uint32_t tuning, const NV_ENC_CONFIG& c) {
  return tuning == num(NV_ENC_TUNING_INFO_LOSSLESS) || (codec == kH264 && c.encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag != 0);
}
bool lossless(const Session& s) { return lossless_of(s.codec, raw(s.init.tuningInfo), s.config); }

int h264_profile_idc(const NV_ENC_CONFIG& c) {
  if (same_guid(c.profileGUID, NV_ENC_H264_PROFILE_BASELINE_GUID)) return 66;
  if (same_guid(c.profileGUID, NV_ENC_H264_PROFILE_MAIN_GUID)) return 77;
  return 100;   // High, and what the automatic choice picks
}

// The H.264 tools an application's configuration asks for.
vgpu_nvenc::H264Options h264_options(const NV_ENC_CONFIG& c, int profile) {
  const NV_ENC_CONFIG_H264& h = c.encodeCodecConfig.h264Config;
  vgpu_nvenc::H264Options o;
  const uint32_t entropy = raw(h.entropyCodingMode), adaptive = raw(h.adaptiveTransformMode);
  o.cabac = profile >= 77 && entropy != num(NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC);
  o.transform8x8 = profile >= 100 && adaptive != num(NV_ENC_H264_ADAPTIVE_TRANSFORM_DISABLE);
  o.slice_mode = static_cast<int>(h.sliceMode <= 3 ? h.sliceMode : 0);
  o.slice_data = static_cast<int>(std::min<uint32_t>(h.sliceModeData, 1u << 20));
  const uint32_t refs = raw(h.numRefL0);
  o.num_ref = refs >= 1 && refs <= 7 ? static_cast<int>(std::min<uint32_t>(refs, 4)) : 2;
  o.max_b = profile >= 77 && c.frameIntervalP > 1 ? std::min(c.frameIntervalP - 1, 8) : 0;
  o.deblock = h.disableDeblockingFilterIDC != 1;
  return o;
}

vgpu_nvenc::HevcOptions hevc_options(const NV_ENC_CONFIG& c) {
  const NV_ENC_CONFIG_HEVC& h = c.encodeCodecConfig.hevcConfig;
  vgpu_nvenc::HevcOptions o;
  const uint32_t refs = raw(h.numRefL0);
  o.num_ref = refs >= 1 && refs <= 7 ? static_cast<int>(std::min<uint32_t>(refs, 4)) : 2;
  o.max_b = c.frameIntervalP > 1 ? std::min(c.frameIntervalP - 1, 8) : 0;
  return o;
}

std::unique_ptr<vgpu_nvenc::VideoEncoder> make_encoder(int codec, const NV_ENC_CONFIG& c, int w, int h, int fn, int fd) {
  if (codec == kHevc) return std::make_unique<vgpu_nvenc::HevcEncoder>(w, h, fn, fd, hevc_options(c));
  const int profile = h264_profile_idc(c);
  return std::make_unique<vgpu_nvenc::H264Encoder>(w, h, fn, fd, profile, h264_options(c, profile));
}

// The parameter sets the encoder of a configuration writes (for the lossless tuning, the PCM writers').
std::vector<uint8_t> parameter_sets_of(int codec, bool is_lossless, const NV_ENC_CONFIG& cfg, int width, int height, uint32_t rate_num, uint32_t rate_den) {
  const int fps_num = rate_num ? static_cast<int>(rate_num) : 30, fps_den = rate_den ? static_cast<int>(rate_den) : 1;
  if (!is_lossless) return make_encoder(codec, cfg, width, height, fps_num, fps_den)->parameter_sets();
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
  return parameter_sets_of(s.codec, lossless(s), s.config, static_cast<int>(s.width), static_cast<int>(s.height), s.init.frameRateNum, s.init.frameRateDen);
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

// What the application's GOP settings ask of the encoder.
struct GopCfg {
  bool intra_only = false;     // frameIntervalP 0: an IDR picture, then I pictures
  bool periodic = false;       // an IDR picture every idr_period pictures
  uint32_t idr_period = 0;
  int nb = 0;                  // B pictures between the P pictures (frameIntervalP - 1), where the encoder has B pictures
  bool repeat_ps = false;      // repeatSPSPPS
};
GopCfg gop_cfg(const Session& s) {
  const NV_ENC_CONFIG& c = s.config;
  GopCfg g;
  uint32_t idr_period;
  if (s.codec == kHevc) {
    idr_period = c.encodeCodecConfig.hevcConfig.idrPeriod ? c.encodeCodecConfig.hevcConfig.idrPeriod : c.gopLength;
    g.repeat_ps = c.encodeCodecConfig.hevcConfig.repeatSPSPPS != 0;
  } else {
    idr_period = c.encodeCodecConfig.h264Config.idrPeriod ? c.encodeCodecConfig.h264Config.idrPeriod : c.gopLength;
    g.repeat_ps = c.encodeCodecConfig.h264Config.repeatSPSPPS != 0;
  }
  g.idr_period = idr_period;
  g.periodic = idr_period != 0 && idr_period != NVENC_INFINITE_GOPLENGTH;
  g.intra_only = c.frameIntervalP == 0;
  if (c.frameIntervalP > 1 && s.enc && s.enc->supports_b()) g.nb = std::min(c.frameIntervalP - 1, 8);   // the encoders code at most eight B pictures between P pictures
  return g;
}

// A coded picture on its way to an output buffer.
struct Packet {
  std::vector<uint8_t> data;
  uint64_t ts = 0, dur = 0;
  int pic_struct = 1;
  uint32_t pic_type = NV_ENC_PIC_TYPE_IDR;
  uint32_t slices = 1;
};

// The picture a call brought: the samples and what the call said about them.
struct Input1 {
  const Frame* frame = nullptr;
  uint64_t ts = 0, dur = 0;
  int pic_struct = 1;
  int disp = 0;
  bool want_ps = false;   // the call asked for the parameter sets with this picture (NV_ENC_PIC_FLAG_OUTPUT_SPSPPS)
};

// A number that changes with every setting the encoder is built from.
uint64_t config_sig(const Session& s) {
  if (s.codec == kHevc) {
    const vgpu_nvenc::HevcOptions o = hevc_options(s.config);
    return 0x4845564300000000ull ^ (static_cast<uint64_t>(o.num_ref) << 8) ^ static_cast<uint64_t>(o.max_b);
  }
  const int profile = h264_profile_idc(s.config);
  const vgpu_nvenc::H264Options o = h264_options(s.config, profile);
  uint64_t v = 1469598103934665603ull;
  for (uint64_t x : {static_cast<uint64_t>(o.cabac), static_cast<uint64_t>(o.transform8x8), static_cast<uint64_t>(o.slice_mode), static_cast<uint64_t>(o.slice_data),
                     static_cast<uint64_t>(o.num_ref), static_cast<uint64_t>(o.max_b), static_cast<uint64_t>(o.deblock)})
    v = (v ^ x) * 1099511628211ull;
  return v;
}

// (Re)creates the encoder when the stream's size, rate, profile or tools changed: a new coded video sequence.
bool ensure_encoder(Session& s, int w, int h, int fn, int fd) {
  const int profile = s.codec == kH264 ? h264_profile_idc(s.config) : 1;
  const uint64_t sig = config_sig(s);
  if (s.enc && s.enc_codec == s.codec && s.enc_w == w && s.enc_h == h && s.enc_fps_num == fn && s.enc_fps_den == fd && s.enc_profile == profile && s.enc_sig == sig) return false;
  s.enc = make_encoder(s.codec, s.config, w, h, fn, fd);
  s.trial.reset();
  s.enc_codec = s.codec;
  s.enc_w = w;
  s.enc_h = h;
  s.enc_fps_num = fn;
  s.enc_fps_den = fd;
  s.enc_profile = profile;
  s.enc_sig = sig;
  s.since_idr = 0;
  s.sent_parameter_sets = false;   // a new sequence: the decoder needs its headers
  return true;
}

// The rate controller's settings, from the application's.
vgpu_nvenc::RateController::Config rc_config(const Session& s, const GopCfg& g, int rate_num, int rate_den) {
  const NV_ENC_RC_PARAMS& rc = s.config.rcParams;
  vgpu_nvenc::RateController::Config c;
  const uint32_t mode = raw(rc.rateControlMode);
  c.bitrate = mode == num(NV_ENC_PARAMS_RC_CONSTQP) ? 0 : static_cast<double>(rc.averageBitRate);
  c.max_bitrate = mode == num(NV_ENC_PARAMS_RC_VBR) ? static_cast<double>(rc.maxBitRate) : 0;
  c.fps = static_cast<double>(rate_num) / std::max(rate_den, 1);
  c.cbr = mode == num(NV_ENC_PARAMS_RC_CBR);
  c.vbv_bits = static_cast<double>(rc.vbvBufferSize);
  c.intra_fraction = g.intra_only ? 1.0 : (g.periodic ? 1.0 / std::max<uint32_t>(g.idr_period, 1) : 0.0);
  c.b_per_p = g.nb;
  for (int t = 0; t < 3; ++t) {
    const NV_ENC_QP& mn = rc.minQP;
    const NV_ENC_QP& mx = rc.maxQP;
    c.qp_min[t] = rc.enableMinQP ? static_cast<int>(t == 0 ? mn.qpIntra : (t == 2 ? mn.qpInterB : mn.qpInterP)) : 0;
    c.qp_max[t] = rc.enableMaxQP ? static_cast<int>(t == 0 ? mx.qpIntra : (t == 2 ? mx.qpInterB : mx.qpInterP)) : 51;
    c.qp_min[t] = std::min(51, std::max(0, c.qp_min[t]));
    c.qp_max[t] = std::min(51, std::max(c.qp_min[t], c.qp_max[t]));
  }
  return c;
}
uint64_t rc_sig_of(const vgpu_nvenc::RateController::Config& c) {
  uint64_t v = 1469598103934665603ull;
  auto mix = [&](uint64_t x) { v = (v ^ x) * 1099511628211ull; };
  mix(static_cast<uint64_t>(c.bitrate));
  mix(static_cast<uint64_t>(c.max_bitrate));
  mix(static_cast<uint64_t>(c.fps * 1000));
  mix(c.cbr);
  mix(static_cast<uint64_t>(c.vbv_bits));
  mix(static_cast<uint64_t>(c.intra_fraction * 1e6));
  mix(static_cast<uint64_t>(c.b_per_p));
  for (int t = 0; t < 3; ++t) {
    mix(static_cast<uint64_t>(c.qp_min[t]));
    mix(static_cast<uint64_t>(c.qp_max[t]));
  }
  return v;
}

// The slice QP of an IDR picture under rate control: a stateless search on a scratch encoder, so that one frame always encodes to the same bytes.
int idr_qp(Session& s, const Frame& f, int rate_num, int rate_den, const GopCfg& g) {
  const NV_ENC_RC_PARAMS& rc = s.config.rcParams;
  const double frame_bits = static_cast<double>(rc.averageBitRate) * rate_den / std::max(rate_num, 1);
  // an I picture takes about four times a P picture's bits; with a short GOP its share of the budget is smaller
  const double gop_len = g.periodic ? static_cast<double>(std::max<uint32_t>(g.idr_period, 1)) : 1e9;
  const double p_bits = frame_bits * gop_len / (gop_len + 3.0);
  const double i_bits = g.intra_only || (g.periodic && g.idr_period <= 1) ? frame_bits : 4.0 * p_bits;
  if (!s.trial) s.trial = make_encoder(s.codec, s.config, f.w, f.h, rate_num, rate_den);
  const vgpu_nvenc::EncPicture pic = to_enc_picture(f);
  auto bits_at = [&](int q) {
    vgpu_nvenc::EncStats st;
    s.trial->encode(pic, vgpu_nvenc::PicType::kIdr, q, &st);
    return static_cast<double>(st.bytes) * 8;
  };
  // Bracket the QP at which the size crosses the target, then bisect between the two sides (at most eight encodes of the scratch encoder).
  int qp = std::min(48, std::max(8, vgpu_nvenc::initial_qp_for(f.w, f.h, rate_num, rate_den, static_cast<long>(i_bits * rate_num / std::max(rate_den, 1)))));
  int lo = -1, hi = 52;   // largest QP known to be too big (more bits than the target), smallest known to be small enough
  double lo_bits = 0, hi_bits = 0;
  int best_qp = qp;
  double best_err = 1e30;
  for (int trial = 0; trial < 8; ++trial) {
    const double bits = bits_at(qp);
    const double err = std::fabs(std::log(bits / i_bits));
    if (err < best_err) {
      best_err = err;
      best_qp = qp;
    }
    if (bits / i_bits > 0.9 && bits / i_bits < 1.12) break;
    if (bits > i_bits) {
      lo = qp;
      lo_bits = bits;
    } else {
      hi = qp;
      hi_bits = bits;
    }
    int next;
    if (lo >= 0 && hi <= 51) {
      if (hi - lo <= 1) break;
      next = lo + (hi - lo) / 2;
    } else {
      next = qp + static_cast<int>(std::lround(6.0 * std::log2(bits / i_bits)));
      if (next == qp) next += bits > i_bits ? 1 : -1;
      next = std::min(51, std::max(8, next));
      if (next == qp) break;
    }
    qp = next;
  }
  (void)lo_bits;
  (void)hi_bits;
  return best_qp;
}

// Codes one picture with the QP the settings and the rate control pick (a second or third attempt when the size misses the rate control's target), updates the
// rate control, and wraps the result with the parameter sets when they are due.
Packet code_picture(Session& s, const Input1& in, vgpu_nvenc::PicType t, int rate_num, int rate_den, const GopCfg& g) {
  using vgpu_nvenc::PicType;
  using vgpu_nvenc::RateController;
  const NV_ENC_RC_PARAMS& rc = s.config.rcParams;
  const uint32_t mode = raw(rc.rateControlMode);
  const Frame& f = *in.frame;
  const bool intra = t == PicType::kIdr || t == PicType::kIntra;
  const vgpu_nvenc::RateController::Config rcc = rc_config(s, g, rate_num, rate_den);
  const uint64_t sig = rc_sig_of(rcc);
  if (sig != s.rc_sig) {
    s.rc.configure(rcc);
    s.rc_sig = sig;
  }
  const RateController::Type rt = intra ? RateController::kIntra : (t == PicType::kBi ? RateController::kB : RateController::kP);
  const vgpu_nvenc::EncPicture pic = to_enc_picture(f);
  auto qp_limits = [&](int q) {
    q = std::max(q, rcc.qp_min[rt]);
    q = std::min(q, rcc.qp_max[rt]);
    return std::min(51, std::max(0, q));
  };
  std::vector<uint8_t> picture;
  int qp;
  auto code = [&](int q) {
    vgpu_nvenc::EncStats stats;
    picture = s.enc->encode_at(pic, t, q, in.disp, t == PicType::kBi ? nullptr : &stats);
    return static_cast<double>(picture.size()) * 8;
  };
  if (mode == num(NV_ENC_PARAMS_RC_CONSTQP) || !s.rc.active()) {
    qp = intra ? rc.constQP.qpIntra : (t == PicType::kBi ? rc.constQP.qpInterB : rc.constQP.qpInterP);
    if (mode != num(NV_ENC_PARAMS_RC_CONSTQP) && rc.targetQuality > 0) qp = rc.targetQuality;   // VBR with no bit rate: constant quality
    qp = qp_limits(qp);
    code(qp);
  } else if (t == PicType::kIdr) {
    qp = qp_limits(idr_qp(s, f, rate_num, rate_den, g));
    const double bits = code(qp);
    s.rc.update(rt, qp, bits);
  } else {
    qp = s.rc.qp_for(rt);
    const double target = s.rc.target_bits(rt);
    double bits = code(qp);
    for (int attempt = 0; attempt < 2; ++attempt) {
      const double ratio = bits / std::max(target, 1.0);
      const bool too_big = bits > s.rc.max_bits();
      if (!too_big && ratio > 0.94 && ratio < 1.06) break;
      const int qp2 = s.rc.next_qp(rt, qp, bits, too_big ? std::min(target, s.rc.max_bits()) : target);
      if (qp2 == qp) break;
      s.enc->rollback();
      const double bits2 = code(qp2);
      s.rc.learn_alpha(qp, bits, qp2, bits2);
      const bool better = std::fabs(std::log(bits2 / target)) <= std::fabs(std::log(bits / target)) && !(bits2 > s.rc.max_bits() && bits <= s.rc.max_bits());
      if (better) {
        qp = qp2;
        bits = bits2;
      } else {
        // the first attempt was the closer one: encode it again to leave the encoder in its state
        s.enc->rollback();
        bits = code(qp);
        break;
      }
    }
    s.rc.update(rt, qp, bits);
    if (std::getenv("VGPU_NVENC_RC_TRACE")) std::fprintf(stderr, "rc: type %d qp %d bits %.0f target %.0f afford %.0f spent %.0f budget %.0f\n", static_cast<int>(rt), qp, bits, target, 0.0, s.rc.spent(), s.rc.budget());
  }
  Packet pk;
  pk.ts = in.ts;
  pk.dur = in.dur;
  pk.pic_struct = in.pic_struct;
  pk.slices = static_cast<uint32_t>(s.enc->last_slices());
  pk.pic_type = t == PicType::kIdr ? num(NV_ENC_PIC_TYPE_IDR)
                : (t == PicType::kIntra ? num(NV_ENC_PIC_TYPE_I) : (t == PicType::kBi ? num(NV_ENC_PIC_TYPE_B) : num(NV_ENC_PIC_TYPE_P)));
  size_t skip = 0;
  if (!s.sent_parameter_sets || in.want_ps || (g.repeat_ps && t == PicType::kIdr)) {
    pk.data = s.enc->parameter_sets();
    // The digest SEI message of an intra picture is the first NAL unit of its access unit, and needs the four-byte start code,
    // only when no parameter sets come before it (Annex B, zero_byte).
    if (picture.size() > 5 && picture[3] == 1 && (s.codec == kH264 ? (picture[4] & 31) == 6 : ((picture[4] >> 1) & 63) == 39)) skip = 1;
  }
  pk.data.insert(pk.data.end(), picture.begin() + static_cast<std::ptrdiff_t>(skip), picture.end());
  s.sent_parameter_sets = true;
  return pk;
}

// Codes the pictures that wait for a P picture, with the last of them as that P picture and the rest as B pictures (end of stream, or an IDR
// or intra picture is due): the packets in coding order.
void flush_waiting(Session& s, int rate_num, int rate_den, const GopCfg& g, std::vector<Packet>* out) {
  if (s.waiting.empty()) return;
  std::vector<Session::Waiting> w = std::move(s.waiting);
  s.waiting.clear();
  Input1 a;
  a.frame = &w.back().frame;
  a.ts = w.back().ts;
  a.dur = w.back().dur;
  a.pic_struct = w.back().pic_struct;
  a.disp = w.back().disp;
  a.want_ps = w.back().want_ps;
  out->push_back(code_picture(s, a, vgpu_nvenc::PicType::kInter, rate_num, rate_den, g));
  for (size_t i = 0; i + 1 < w.size(); ++i) {
    Input1 b;
    b.frame = &w[i].frame;
    b.ts = w[i].ts;
    b.dur = w[i].dur;
    b.pic_struct = w[i].pic_struct;
    b.disp = w[i].disp;
    b.want_ps = w[i].want_ps;
    out->push_back(code_picture(s, b, vgpu_nvenc::PicType::kBi, rate_num, rate_den, g));
  }
}

// Hands coded pictures to output buffers: the buffers of the calls that returned NV_ENC_ERR_NEED_MORE_INPUT, in the order of the calls, then
// the current call's (the card fills them in this order whatever the picture; pictures arrive in coding order).
void deliver(Session& s, std::vector<Packet>& packets, void* current_out) {
  std::vector<void*> bufs(s.outstanding.begin(), s.outstanding.end());
  s.outstanding.clear();
  if (current_out) bufs.push_back(current_out);
  for (size_t i = 0; i < packets.size() && i < bufs.size(); ++i) {
    auto it = s.outputs.find(bufs[i]);
    if (it == s.outputs.end()) continue;
    Packet& p = packets[i];
    it->second->pending.push_back({std::move(p.data), p.ts, p.dur, static_cast<uint32_t>(s.frames++), p.pic_struct, p.pic_type, p.slices});
  }
}

// One picture of a compressing session: decides its type from the GOP settings and the application's flags, and codes it, or holds it for the
// P picture that follows it in display order (B pictures: the card answers NV_ENC_ERR_NEED_MORE_INPUT and fills the output buffers when the
// group is coded). Returns the status NvEncEncodePicture answers.
NVENCSTATUS encode_compressed(Session& s, const Frame& f, const NV_ENC_PIC_PARAMS& pp, int pic_struct, void* out_handle, int rate_num, int rate_den) {
  std::vector<Packet> packets;
  const bool changed_dims = s.enc && (s.enc_w != f.w || s.enc_h != f.h);
  if (changed_dims && !s.waiting.empty()) {
    GopCfg g0 = gop_cfg(s);
    flush_waiting(s, s.enc_fps_num, s.enc_fps_den, g0, &packets);
  }
  ensure_encoder(s, f.w, f.h, rate_num, rate_den);
  const GopCfg g = gop_cfg(s);
  const uint32_t flags = pp.encodePicFlags;
  Input1 in;
  in.frame = &f;
  in.ts = pp.inputTimeStamp;
  in.dur = pp.inputDuration;
  in.pic_struct = pic_struct;
  in.want_ps = (flags & NV_ENC_PIC_FLAG_OUTPUT_SPSPPS) != 0;
  using vgpu_nvenc::PicType;
  PicType t;
  bool idr_now;
  if (s.init.enablePTD) {
    idr_now = s.since_idr == 0 || s.force_idr_next || (flags & NV_ENC_PIC_FLAG_FORCEIDR) || (g.periodic && s.since_idr >= g.idr_period);
    if (idr_now) t = PicType::kIdr;
    else if (g.intra_only || (flags & NV_ENC_PIC_FLAG_FORCEINTRA)) t = PicType::kIntra;   // frameIntervalP 0: the card writes an IDR picture and then I pictures (measured)
    else t = PicType::kInter;
  } else {
    // the application decides (NV_ENC_PIC_PARAMS::pictureType); a first picture is always an IDR
    const uint32_t pt = raw(pp.pictureType);
    idr_now = s.since_idr == 0 || pt == num(NV_ENC_PIC_TYPE_IDR);
    t = idr_now ? PicType::kIdr : (pt == num(NV_ENC_PIC_TYPE_I) ? PicType::kIntra : PicType::kInter);
  }
  s.force_idr_next = false;
  if (t != PicType::kInter || g.nb == 0 || !s.init.enablePTD) {
    // coded now: an IDR or intra picture first flushes the pictures that wait for their P picture
    flush_waiting(s, rate_num, rate_den, g, &packets);
    if (t == PicType::kIdr) s.since_idr = 0;
    in.disp = static_cast<int>(s.since_idr);
    packets.push_back(code_picture(s, in, t, rate_num, rate_den, g));
    s.since_idr = s.since_idr + 1;
    deliver(s, packets, out_handle);
    return NV_ENC_SUCCESS;
  }
  // a P picture position with B pictures: the P picture ends a group when enough pictures wait for it, or when the GOP ends with it
  const int d = static_cast<int>(s.since_idr);
  const bool last_in_gop = g.periodic && static_cast<uint32_t>(d) + 1 == g.idr_period;
  if (static_cast<int>(s.waiting.size()) < g.nb && !last_in_gop) {
    Session::Waiting w;
    w.frame = f;
    w.ts = in.ts;
    w.dur = in.dur;
    w.pic_struct = pic_struct;
    w.disp = d;
    w.want_ps = in.want_ps;
    s.waiting.push_back(std::move(w));
    s.outstanding.push_back(out_handle);
    s.since_idr = s.since_idr + 1;
    return NV_ENC_ERR_NEED_MORE_INPUT;
  }
  in.disp = d;
  packets.push_back(code_picture(s, in, PicType::kInter, rate_num, rate_den, g));
  {
    std::vector<Session::Waiting> w = std::move(s.waiting);
    s.waiting.clear();
    for (const Session::Waiting& b : w) {
      Input1 bi;
      bi.frame = &b.frame;
      bi.ts = b.ts;
      bi.dur = b.dur;
      bi.pic_struct = b.pic_struct;
      bi.disp = b.disp;
      bi.want_ps = b.want_ps;
      packets.push_back(code_picture(s, bi, PicType::kBi, rate_num, rate_den, g));
    }
  }
  s.since_idr = s.since_idr + 1;
  deliver(s, packets, out_handle);
  return NV_ENC_SUCCESS;
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
  if (params->encodePicFlags & NV_ENC_PIC_FLAG_EOS) {
    // end of stream: the pictures that wait for a P picture are coded now, into the buffers of their calls
    if (!s->waiting.empty() && s->enc) {
      std::vector<Packet> packets;
      flush_waiting(*s, s->enc_fps_num, s->enc_fps_den, gop_cfg(*s), &packets);
      deliver(*s, packets, nullptr);
    }
    return NV_ENC_SUCCESS;
  }
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
  Output& o = *out->second;
  if (lossless(*s)) {
    // the PCM streams: every picture is an IDR picture
    std::vector<uint8_t> bits, picture;
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
      vgpu_nvenc::H264Stream st;
      st.width = w;
      st.height = h;
      st.fps_num = rate_num;
      st.fps_den = rate_den;
      if (!s->sent_parameter_sets || (params->encodePicFlags & NV_ENC_PIC_FLAG_OUTPUT_SPSPPS) || s->config.encodeCodecConfig.h264Config.repeatSPSPPS)
        bits = st.parameter_sets();
      picture = st.idr(luma, chroma, 0);
    }
    s->sent_parameter_sets = true;
    bits.insert(bits.end(), picture.begin(), picture.end());
    o.pending.push_back({std::move(bits), params->inputTimeStamp, params->inputDuration, static_cast<uint32_t>(s->frames++), ps, num(NV_ENC_PIC_TYPE_IDR), 1});
    return NV_ENC_SUCCESS;
  }
  return encode_compressed(*s, f, *params, ps, params->outputBitstream, rate_num, rate_den);
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
  params->numSlices = o.current.slices;
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
  NV_ENC_PRESET_CONFIG pc{};
  pc.version = NV_ENC_PRESET_CONFIG_VER;
  pc.presetCfg.version = NV_ENC_CONFIG_VER;
  const int tuning = static_cast<int>(raw(init->tuningInfo));
  if (init->encodeConfig) pc.presetCfg = *init->encodeConfig;
  else preset_config(codec, std::max(1, preset_index(init->presetGUID)), tuning >= 1 && tuning <= 4 ? tuning : 1, &pc);
  const std::vector<uint8_t> ps = parameter_sets_of(codec, lossless_of(codec, static_cast<uint32_t>(tuning), pc.presetCfg), pc.presetCfg,
                                                    static_cast<int>(init->encodeWidth), static_cast<int>(init->encodeHeight), init->frameRateNum, init->frameRateDen);
  if (params->inBufferSize < ps.size()) return NV_ENC_ERR_OUT_OF_MEMORY;
  std::memcpy(params->spsppsBuffer, ps.data(), ps.size());
  *params->outSPSPPSPayloadSize = static_cast<uint32_t>(ps.size());
  return NV_ENC_SUCCESS;
}
