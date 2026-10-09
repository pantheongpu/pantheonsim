// NVENC's API as an application calls it, against what NVIDIA's libnvidia-encode
// answered on an RTX 3060 (driver 595, NVENC API 13.0): the function table, session
// opening, every query (codecs, profiles, input formats, capabilities, presets
// and their configurations), initialisation with its error strings, input
// buffers and registered resources, encoding and the fields of a locked
// bitstream, sequence parameters, statistics and the rest of the table.
//
// The program prints one line per fact; nvidia/tests/e2e/nvenc_api.rtx3060.txt is
// what the card printed, and run_nvenc.sh compares the simulator's output with
// it (and, with --card, regenerates it from the driver's library).
//
// What it leaves out, because the simulator is not NVIDIA's encoder: the bytes
// of a bitstream and its size, and the picture types after the first (the
// simulator writes every frame as an IDR; the card's P frames are not
// reproduced).
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <nvEncodeAPI.h>

static NV_ENCODE_API_FUNCTION_LIST f{};
static void* enc = nullptr;

static const char* name_of(const GUID& g) {
  static char buf[64];
#define G(N) \
  if (!std::memcmp(&g, &N, sizeof g)) return #N;
  G(NV_ENC_CODEC_H264_GUID) G(NV_ENC_CODEC_HEVC_GUID) G(NV_ENC_CODEC_AV1_GUID)
  G(NV_ENC_PRESET_P1_GUID) G(NV_ENC_PRESET_P2_GUID) G(NV_ENC_PRESET_P3_GUID) G(NV_ENC_PRESET_P4_GUID)
  G(NV_ENC_PRESET_P5_GUID) G(NV_ENC_PRESET_P6_GUID) G(NV_ENC_PRESET_P7_GUID)
  G(NV_ENC_H264_PROFILE_BASELINE_GUID) G(NV_ENC_H264_PROFILE_MAIN_GUID) G(NV_ENC_H264_PROFILE_HIGH_GUID)
  G(NV_ENC_H264_PROFILE_HIGH_444_GUID) G(NV_ENC_H264_PROFILE_STEREO_GUID)
  G(NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID) G(NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID)
  G(NV_ENC_HEVC_PROFILE_MAIN_GUID) G(NV_ENC_HEVC_PROFILE_MAIN10_GUID) G(NV_ENC_HEVC_PROFILE_FREXT_GUID)
  G(NV_ENC_AV1_PROFILE_MAIN_GUID) G(NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID)
#undef G
  std::snprintf(buf, sizeof buf, "%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g.Data1, g.Data2, g.Data3, g.Data4[0], g.Data4[1],
                g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);
  return buf;
}

static uint64_t fnv(const void* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
  return h;
}

// Store a number into an enum field the API defines no name for: the card accepts or
// refuses those by value, and loading one back as its enum type is undefined, so it
// is written byte for byte and never read as the enum.
template <class E>
static void poke(E& field, uint32_t v) {
  static_assert(sizeof(E) == sizeof v, "a 32-bit enum");
  std::memcpy(&field, &v, sizeof v);
}

static void last_error(const char* what) {
  const char* e = f.nvEncGetLastErrorString ? f.nvEncGetLastErrorString(enc) : nullptr;
  std::printf("  last error after %s: \"%s\"\n", what, e ? e : "(null)");
}

static bool open_session(void** out, int device_type = NV_ENC_DEVICE_TYPE_CUDA) {
  CUcontext ctx = nullptr;
  cuCtxGetCurrent(&ctx);
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op{};
  op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  poke(op.deviceType, static_cast<uint32_t>(device_type));
  op.device = ctx;
  op.apiVersion = NVENCAPI_VERSION;
  return f.nvEncOpenEncodeSessionEx(&op, out) == NV_ENC_SUCCESS;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  (void)argc;
  (void)argv;
  void* lib = dlopen("libnvidia-encode.so.1", RTLD_NOW);
  if (!lib) {
    std::printf("SKIP: no libnvidia-encode.so.1\n");
    return 0;
  }
  using GetVer = NVENCSTATUS (*)(uint32_t*);
  using Create = NVENCSTATUS (*)(NV_ENCODE_API_FUNCTION_LIST*);
  auto get_version = reinterpret_cast<GetVer>(dlsym(lib, "NvEncodeAPIGetMaxSupportedVersion"));
  auto create = reinterpret_cast<Create>(dlsym(lib, "NvEncodeAPICreateInstance"));
  cudaFree(nullptr);

  // ---- version and function table -------------------------------------------
  uint32_t ver = 0;
  {
    const int r = get_version(&ver);
    std::printf("GetMaxSupportedVersion: %d, %u.%u\n", r, ver >> 4, ver & 15);
  }
  std::printf("GetMaxSupportedVersion(null): %d\n", get_version(nullptr));
  std::printf("CreateInstance(null): %d\n", create(nullptr));
  {
    NV_ENCODE_API_FUNCTION_LIST bad{};
    bad.version = NV_ENCODE_API_FUNCTION_LIST_VER + 1;
    std::printf("CreateInstance(version+1): %d\n", create(&bad));
    bad = {};
    bad.version = 0;
    std::printf("CreateInstance(version 0): %d\n", create(&bad));
  }
  f.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  std::printf("CreateInstance: %d\n", create(&f));
  const char* names[] = {"OpenEncodeSession", "GetEncodeGUIDCount", "GetEncodeProfileGUIDCount", "GetEncodeProfileGUIDs", "GetEncodeGUIDs",
                         "GetInputFormatCount", "GetInputFormats", "GetEncodeCaps", "GetEncodePresetCount", "GetEncodePresetGUIDs",
                         "GetEncodePresetConfig", "InitializeEncoder", "CreateInputBuffer", "DestroyInputBuffer", "CreateBitstreamBuffer",
                         "DestroyBitstreamBuffer", "EncodePicture", "LockBitstream", "UnlockBitstream", "LockInputBuffer",
                         "UnlockInputBuffer", "GetEncodeStats", "GetSequenceParams", "RegisterAsyncEvent", "UnregisterAsyncEvent",
                         "MapInputResource", "UnmapInputResource", "DestroyEncoder", "InvalidateRefFrames", "OpenEncodeSessionEx",
                         "RegisterResource", "UnregisterResource", "ReconfigureEncoder", "reserved1", "CreateMVBuffer", "DestroyMVBuffer",
                         "RunMotionEstimationOnly", "GetLastErrorString", "SetIOCudaStreams", "GetEncodePresetConfigEx",
                         "GetSequenceParamEx", "RestoreEncoderState", "LookaheadPicture"};
  {
    void** p = reinterpret_cast<void**>(&f.nvEncOpenEncodeSession);
    std::string missing;
    for (int i = 0; i < 43; ++i)
      if (!p[i] && i != 33) missing += std::string(" ") + names[i];
    std::printf("function table: slots missing:%s\n", missing.empty() ? " none" : missing.c_str());
  }

  // ---- opening a session ----------------------------------------------------
  {
    // A session that opens when it should not is closed again: the driver
    // limits how many an application holds.
    void* e = nullptr;
    const auto attempt = [&](const char* what, NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS* params, void** out) {
      e = nullptr;
      const int r = f.nvEncOpenEncodeSessionEx(params, out);
      std::printf("OpenEncodeSessionEx(%s): %d\n", what, r);
      if (r == NV_ENC_SUCCESS && e) f.nvEncDestroyEncoder(e);
    };
    attempt("null params", nullptr, &e);
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op{};
    op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    op.apiVersion = NVENCAPI_VERSION;
    attempt("null encoder out", &op, nullptr);
    attempt("null device", &op, &e);
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    op.device = ctx;
    op.version = 0;
    attempt("version 0", &op, &e);
    op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    op.apiVersion = NVENCAPI_VERSION + 1;
    attempt("api version too new", &op, &e);
    op.apiVersion = 0;
    attempt("api version 0", &op, &e);
    op.apiVersion = NVENCAPI_VERSION;
    for (int dt : {static_cast<int>(NV_ENC_DEVICE_TYPE_DIRECTX), static_cast<int>(NV_ENC_DEVICE_TYPE_OPENGL), 7}) {
      poke(op.deviceType, static_cast<uint32_t>(dt));
      attempt(("device type " + std::to_string(dt)).c_str(), &op, &e);
    }
    op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    op.device = ctx;
    std::printf("OpenEncodeSessionEx: %d\n", f.nvEncOpenEncodeSessionEx(&op, &enc));
  }
  if (!enc) return 1;
  last_error("open");

  // ---- codecs, profiles, input formats, capabilities, presets ---------------
  uint32_t n = 0;
  {
    int r = f.nvEncGetEncodeGUIDCount(enc, &n);
    std::printf("GetEncodeGUIDCount: %d, %u\n", r, n);
    std::vector<GUID> guids(n);
    uint32_t got = 0;
    r = f.nvEncGetEncodeGUIDs(enc, guids.data(), n, &got);
    std::printf("GetEncodeGUIDs: %d, %u:", r, got);
    for (uint32_t i = 0; i < got; ++i) std::printf(" %s", name_of(guids[i]));
    std::printf("\n");
    std::printf("GetEncodeGUIDs(room for 1): %d\n", f.nvEncGetEncodeGUIDs(enc, guids.data(), 1, &got));
    std::printf("  got %u\n", got);
    std::printf("GetEncodeGUIDCount(null): %d\n", f.nvEncGetEncodeGUIDCount(enc, nullptr));
    std::printf("GetEncodeGUIDCount(null session): %d\n", f.nvEncGetEncodeGUIDCount(nullptr, &n));
  }
  const GUID codecs[] = {NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID, NV_ENC_CODEC_AV1_GUID};
  const char* codec_names[] = {"H264", "HEVC", "AV1"};
  const GUID presets[] = {NV_ENC_PRESET_P1_GUID, NV_ENC_PRESET_P2_GUID, NV_ENC_PRESET_P3_GUID, NV_ENC_PRESET_P4_GUID,
                          NV_ENC_PRESET_P5_GUID, NV_ENC_PRESET_P6_GUID, NV_ENC_PRESET_P7_GUID};
  const NV_ENC_TUNING_INFO tunings[] = {NV_ENC_TUNING_INFO_HIGH_QUALITY, NV_ENC_TUNING_INFO_LOW_LATENCY,
                                        NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, NV_ENC_TUNING_INFO_LOSSLESS};
  for (int c = 0; c < 3; ++c) {
    const GUID& cg = codecs[c];
    std::printf("== %s\n", codec_names[c]);
    uint32_t pc = 0, got = 0;
    int r = f.nvEncGetEncodeProfileGUIDCount(enc, cg, &pc);
    std::printf("profiles: count %d, %u", r, pc);
    std::vector<GUID> pg(pc + 1);
    r = f.nvEncGetEncodeProfileGUIDs(enc, cg, pg.data(), pc, &got);
    std::printf("; list %d, %u:", r, got);
    for (uint32_t i = 0; i < got; ++i) std::printf(" %s", name_of(pg[i]));
    std::printf("\n");
    uint32_t fc = 0;
    r = f.nvEncGetInputFormatCount(enc, cg, &fc);
    std::vector<NV_ENC_BUFFER_FORMAT> fm(fc + 1);
    uint32_t fgot = 0;
    int r2 = f.nvEncGetInputFormats(enc, cg, fm.data(), fc, &fgot);
    std::printf("input formats: %d/%d, %u:", r, r2, fgot);
    for (uint32_t i = 0; i < fgot; ++i) std::printf(" 0x%x", fm[i]);
    std::printf("\n");
    std::printf("caps:");
    for (int cap = 0; cap < 70; ++cap) {
      NV_ENC_CAPS_PARAM cp{};
      cp.version = NV_ENC_CAPS_PARAM_VER;
      poke(cp.capsToQuery, static_cast<uint32_t>(cap));
      int v = -12345;
      const int cs = f.nvEncGetEncodeCaps(enc, cg, &cp, &v);
      if (cs)
        std::printf(" %d:st%d", cap, cs);
      else
        std::printf(" %d:%d", cap, v);
    }
    std::printf("\n");
    uint32_t prc = 0, prgot = 0;
    r = f.nvEncGetEncodePresetCount(enc, cg, &prc);
    std::vector<GUID> prg(prc + 1);
    r2 = f.nvEncGetEncodePresetGUIDs(enc, cg, prg.data(), prc, &prgot);
    std::printf("presets: %d/%d, %u:", r, r2, prgot);
    for (uint32_t i = 0; i < prgot; ++i) std::printf(" %s", name_of(prg[i]));
    std::printf("\n");
    // The configurations: a checksum of each, and the fields most programs read.
    for (int pi = 0; pi < 7; ++pi)
      for (int ti = 0; ti < 4; ++ti) {
        NV_ENC_PRESET_CONFIG pcfg{};
        pcfg.version = NV_ENC_PRESET_CONFIG_VER;
        pcfg.presetCfg.version = NV_ENC_CONFIG_VER;
        r = f.nvEncGetEncodePresetConfigEx(enc, cg, presets[pi], tunings[ti], &pcfg);
        if (r) {
          std::printf("preset p%d tuning %d: status %d\n", pi + 1, static_cast<int>(tunings[ti]), r);
          continue;
        }
        std::printf("preset p%d tuning %d: hash %016llx gop %u P %d rc %d qp %u/%u/%u maxbitrate %u\n", pi + 1, static_cast<int>(tunings[ti]),
                    static_cast<unsigned long long>(fnv(&pcfg, sizeof pcfg)), pcfg.presetCfg.gopLength, pcfg.presetCfg.frameIntervalP,
                    static_cast<int>(pcfg.presetCfg.rcParams.rateControlMode), pcfg.presetCfg.rcParams.constQP.qpInterP,
                    pcfg.presetCfg.rcParams.constQP.qpInterB, pcfg.presetCfg.rcParams.constQP.qpIntra, pcfg.presetCfg.rcParams.maxBitRate);
      }
    NV_ENC_PRESET_CONFIG pcfg{};
    pcfg.version = NV_ENC_PRESET_CONFIG_VER;
    pcfg.presetCfg.version = NV_ENC_CONFIG_VER;
    std::printf("GetEncodePresetConfig (legacy form): %d\n", f.nvEncGetEncodePresetConfig(enc, cg, presets[3], &pcfg));
    std::printf("GetEncodePresetConfigEx(bad preset guid): %d\n",
                f.nvEncGetEncodePresetConfigEx(enc, cg, NV_ENC_CODEC_H264_GUID, NV_ENC_TUNING_INFO_HIGH_QUALITY, &pcfg));
    std::printf("GetEncodePresetConfigEx(tuning 0): %d\n", f.nvEncGetEncodePresetConfigEx(enc, cg, presets[3], NV_ENC_TUNING_INFO_UNDEFINED, &pcfg));
    std::printf("GetEncodePresetConfigEx(tuning 9): %d\n",
                f.nvEncGetEncodePresetConfigEx(enc, cg, presets[3], static_cast<NV_ENC_TUNING_INFO>(9), &pcfg));
    pcfg.version = 0;
    std::printf("GetEncodePresetConfigEx(version 0): %d\n", f.nvEncGetEncodePresetConfigEx(enc, cg, presets[3], tunings[0], &pcfg));
    std::printf("GetEncodePresetConfigEx(null): %d\n", f.nvEncGetEncodePresetConfigEx(enc, cg, presets[3], tunings[0], nullptr));
  }
  {
    GUID junk = {1, 2, 3, {4, 5, 6, 7, 8, 9, 10, 11}};
    uint32_t c = 77;
    std::printf("GetEncodeProfileGUIDCount(unknown codec): %d, %u\n", f.nvEncGetEncodeProfileGUIDCount(enc, junk, &c), c);
    std::printf("GetInputFormatCount(unknown codec): %d\n", f.nvEncGetInputFormatCount(enc, junk, &c));
    std::printf("GetEncodePresetCount(unknown codec): %d\n", f.nvEncGetEncodePresetCount(enc, junk, &c));
    NV_ENC_CAPS_PARAM cp{};
    cp.version = NV_ENC_CAPS_PARAM_VER;
    int v = 0;
    std::printf("GetEncodeCaps(unknown codec): %d\n", f.nvEncGetEncodeCaps(enc, junk, &cp, &v));
    cp.version = 0;
    std::printf("GetEncodeCaps(version 0): %d\n", f.nvEncGetEncodeCaps(enc, codecs[0], &cp, &v));
    cp.version = NV_ENC_CAPS_PARAM_VER;
    std::printf("GetEncodeCaps(null value): %d\n", f.nvEncGetEncodeCaps(enc, codecs[0], &cp, nullptr));
    std::printf("GetEncodeCaps(null param): %d\n", f.nvEncGetEncodeCaps(enc, codecs[0], nullptr, &v));
  }
  f.nvEncDestroyEncoder(enc);
  enc = nullptr;

  // ---- initialisation -------------------------------------------------------
  const auto session = [&]() {
    void* e = nullptr;
    open_session(&e);
    return e;
  };
  const auto init_params = [&](NV_ENC_INITIALIZE_PARAMS* ip, NV_ENC_PRESET_CONFIG* pc, const GUID& codec, uint32_t w, uint32_t h) {
    *ip = {};
    ip->version = NV_ENC_INITIALIZE_PARAMS_VER;
    ip->encodeGUID = codec;
    ip->presetGUID = NV_ENC_PRESET_P4_GUID;
    ip->encodeWidth = w;
    ip->encodeHeight = h;
    ip->darWidth = w;
    ip->darHeight = h;
    ip->frameRateNum = 30;
    ip->frameRateDen = 1;
    ip->enablePTD = 1;
    ip->tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
    *pc = {};
    pc->version = NV_ENC_PRESET_CONFIG_VER;
    pc->presetCfg.version = NV_ENC_CONFIG_VER;
    f.nvEncGetEncodePresetConfigEx(enc, codec, ip->presetGUID, ip->tuningInfo, pc);
    ip->encodeConfig = &pc->presetCfg;
  };
  {
    NV_ENC_INITIALIZE_PARAMS ip;
    NV_ENC_PRESET_CONFIG pc;
    enc = session();
    std::printf("InitializeEncoder(null): %d\n", f.nvEncInitializeEncoder(enc, nullptr));
    init_params(&ip, &pc, NV_ENC_CODEC_H264_GUID, 192, 128);
    ip.version = 0;
    std::printf("InitializeEncoder(version 0): %d\n", f.nvEncInitializeEncoder(enc, &ip));
    init_params(&ip, &pc, NV_ENC_CODEC_H264_GUID, 192, 128);
    pc.presetCfg.version = 0;
    std::printf("InitializeEncoder(config version 0): %d\n", f.nvEncInitializeEncoder(enc, &ip));
    last_error("config version 0");
    f.nvEncDestroyEncoder(enc);

    const struct {
      const char* what;
      GUID codec;
      uint32_t w, h;
    } sizes[] = {{"h264 0x0", NV_ENC_CODEC_H264_GUID, 0, 0},       {"h264 144x48", NV_ENC_CODEC_H264_GUID, 144, 48},
                 {"h264 145x49", NV_ENC_CODEC_H264_GUID, 145, 49}, {"h264 146x50", NV_ENC_CODEC_H264_GUID, 146, 50},
                 {"h264 192x128", NV_ENC_CODEC_H264_GUID, 192, 128}, {"h264 193x129", NV_ENC_CODEC_H264_GUID, 193, 129},
                 {"h264 4096x2304", NV_ENC_CODEC_H264_GUID, 4096, 2304}, {"h264 4097x2304", NV_ENC_CODEC_H264_GUID, 4097, 2304},
                 {"h264 4096x4097", NV_ENC_CODEC_H264_GUID, 4096, 4097}, {"hevc 128x64", NV_ENC_CODEC_HEVC_GUID, 128, 64},
                 {"hevc 129x65", NV_ENC_CODEC_HEVC_GUID, 129, 65},       {"hevc 192x128", NV_ENC_CODEC_HEVC_GUID, 192, 128},
                 {"hevc 8192x8192", NV_ENC_CODEC_HEVC_GUID, 8192, 8192}, {"hevc 8193x4096", NV_ENC_CODEC_HEVC_GUID, 8193, 4096},
                 {"av1 192x128", NV_ENC_CODEC_AV1_GUID, 192, 128},
                 {"h264 145x48", NV_ENC_CODEC_H264_GUID, 145, 48},     {"h264 144x49", NV_ENC_CODEC_H264_GUID, 144, 49},
                 {"h264 4096x49", NV_ENC_CODEC_H264_GUID, 4096, 49},   {"h264 145x4096", NV_ENC_CODEC_H264_GUID, 145, 4096},
                 {"h264 4096x4096", NV_ENC_CODEC_H264_GUID, 4096, 4096}, {"h264 145x4097", NV_ENC_CODEC_H264_GUID, 145, 4097},
                 {"h264 16x16", NV_ENC_CODEC_H264_GUID, 16, 16},       {"h264 1x1", NV_ENC_CODEC_H264_GUID, 1, 1},
                 {"hevc 129x32", NV_ENC_CODEC_HEVC_GUID, 129, 32},     {"hevc 128x65", NV_ENC_CODEC_HEVC_GUID, 128, 65},
                 {"hevc 129x33", NV_ENC_CODEC_HEVC_GUID, 129, 33},     {"hevc 8192x129", NV_ENC_CODEC_HEVC_GUID, 8192, 129},
                 {"hevc 129x8192", NV_ENC_CODEC_HEVC_GUID, 129, 8192}, {"hevc 129x8193", NV_ENC_CODEC_HEVC_GUID, 129, 8193}};
    for (const auto& z : sizes) {
      enc = session();
      init_params(&ip, &pc, z.codec, z.w, z.h);
      const int r = f.nvEncInitializeEncoder(enc, &ip);
      std::printf("InitializeEncoder(%s): %d\n", z.what, r);
      if (r) last_error(z.what);
      f.nvEncDestroyEncoder(enc);
    }
    // Other parameters.
    const auto trial = [&](const char* what, void (*mutate)(NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG*)) {
      enc = session();
      init_params(&ip, &pc, NV_ENC_CODEC_H264_GUID, 192, 128);
      mutate(&ip, &pc);
      const int r = f.nvEncInitializeEncoder(enc, &ip);
      std::printf("InitializeEncoder(%s): %d\n", what, r);
      if (r) last_error(what);
      f.nvEncDestroyEncoder(enc);
    };
    trial("no encode config", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->encodeConfig = nullptr; });
    trial("zero codec guid", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->encodeGUID = GUID{}; });
    trial("zero preset guid", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->presetGUID = GUID{}; });
    trial("codec guid used as preset", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->presetGUID = NV_ENC_CODEC_H264_GUID; });
    trial("zero frame rate", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->frameRateNum = 0; });
    trial("zero frame rate denominator", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->frameRateDen = 0; });
    trial("zero aspect ratio", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->darWidth = i->darHeight = 0; });
    trial("PTD off", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->enablePTD = 0; });
    trial("max size below size", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->maxEncodeWidth = 64; i->maxEncodeHeight = 64; });
    trial("max size above size", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->maxEncodeWidth = 256; i->maxEncodeHeight = 256; });
    trial("async mode", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->enableEncodeAsync = 1; });
    trial("tuning 0", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->tuningInfo = NV_ENC_TUNING_INFO_UNDEFINED; });
    trial("tuning lossless", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { i->tuningInfo = NV_ENC_TUNING_INFO_LOSSLESS; });
    trial("tuning 9", [](NV_ENC_INITIALIZE_PARAMS* i, NV_ENC_PRESET_CONFIG*) { poke(i->tuningInfo, 9); });
    trial("unknown profile guid", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.profileGUID = GUID{1, 2, 3, {4, 5, 6, 7, 8, 9, 10, 11}}; });
    trial("hevc profile for h264", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.profileGUID = NV_ENC_HEVC_PROFILE_MAIN_GUID; });
    trial("baseline profile", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID; });
    trial("gop length 0", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.gopLength = 0; });
    trial("frameIntervalP 9", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.frameIntervalP = 9; });
    trial("gop 5 with frameIntervalP 9", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.gopLength = 5; c->presetCfg.frameIntervalP = 9; });
    trial("gop 9 with frameIntervalP 9", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.gopLength = 9; c->presetCfg.frameIntervalP = 9; });
    trial("gop 1 with frameIntervalP 2", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.gopLength = 1; c->presetCfg.frameIntervalP = 2; });
    trial("gop 2 with frameIntervalP 2", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.gopLength = 2; c->presetCfg.frameIntervalP = 2; });
    trial("gop 0 with frameIntervalP 5", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { c->presetCfg.gopLength = 0; c->presetCfg.frameIntervalP = 5; });
    trial("rate control mode 99", [](NV_ENC_INITIALIZE_PARAMS*, NV_ENC_PRESET_CONFIG* c) { poke(c->presetCfg.rcParams.rateControlMode, 99); });
    // Initialising twice.
    enc = session();
    init_params(&ip, &pc, NV_ENC_CODEC_H264_GUID, 192, 128);
    std::printf("InitializeEncoder: %d\n", f.nvEncInitializeEncoder(enc, &ip));
    std::printf("InitializeEncoder (again): %d\n", f.nvEncInitializeEncoder(enc, &ip));
    f.nvEncDestroyEncoder(enc);
    std::printf("DestroyEncoder(null): %d\n", f.nvEncDestroyEncoder(nullptr));
  }

  // ---- buffers, encoding, the locked bitstream ------------------------------
  {
    const uint32_t W = 192, H = 128;
    NV_ENC_INITIALIZE_PARAMS ip;
    NV_ENC_PRESET_CONFIG pc;
    enc = session();
    // Before initialisation.
    {
      NV_ENC_CREATE_INPUT_BUFFER ib{};
      ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
      ib.width = W;
      ib.height = H;
      ib.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
      std::printf("CreateInputBuffer (before init): %d\n", f.nvEncCreateInputBuffer(enc, &ib));
      if (ib.inputBuffer) f.nvEncDestroyInputBuffer(enc, ib.inputBuffer);
      NV_ENC_CREATE_BITSTREAM_BUFFER ob{};
      ob.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
      std::printf("CreateBitstreamBuffer (before init): %d\n", f.nvEncCreateBitstreamBuffer(enc, &ob));
      if (ob.bitstreamBuffer) f.nvEncDestroyBitstreamBuffer(enc, ob.bitstreamBuffer);
      uint8_t buf[256];
      uint32_t payload = 0;
      NV_ENC_SEQUENCE_PARAM_PAYLOAD sp{};
      sp.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
      sp.inBufferSize = sizeof buf;
      sp.spsppsBuffer = buf;
      sp.outSPSPPSPayloadSize = &payload;
      std::printf("GetSequenceParams (before init): %d\n", f.nvEncGetSequenceParams(enc, &sp));
      NV_ENC_PIC_PARAMS pp{};
      pp.version = NV_ENC_PIC_PARAMS_VER;
      std::printf("EncodePicture (before init): %d\n", f.nvEncEncodePicture(enc, &pp));
    }
    init_params(&ip, &pc, NV_ENC_CODEC_H264_GUID, W, H);
    std::printf("InitializeEncoder: %d\n", f.nvEncInitializeEncoder(enc, &ip));

    // Input buffers: the pitch the driver chooses, per format and width.
    const struct {
      const char* name;
      uint32_t fmt;
    } formats[] = {{"NV12", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_NV12)},       {"YV12", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_YV12)},
                   {"IYUV", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_IYUV)},       {"YUV444", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_YUV444)},
                   {"P010", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_YUV420_10BIT)}, {"444-10", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_YUV444_10BIT)},
                   {"ARGB", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_ARGB)},       {"ARGB10", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_ARGB10)},
                   {"AYUV", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_AYUV)},       {"ABGR", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_ABGR)},
                   {"ABGR10", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_ABGR10)},   {"U8", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_U8)},
                   {"undefined", static_cast<uint32_t>(NV_ENC_BUFFER_FORMAT_UNDEFINED)}, {"bogus", 7u}};
    for (const auto& fm : formats)
      for (uint32_t w : {W, uint32_t(200)}) {
        NV_ENC_CREATE_INPUT_BUFFER ib{};
        ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
        ib.width = w;
        ib.height = H;
        poke(ib.bufferFmt, fm.fmt);
        const int r = f.nvEncCreateInputBuffer(enc, &ib);
        uint32_t pitch = 0;
        int lr = -1;
        if (!r) {
          NV_ENC_LOCK_INPUT_BUFFER lk{};
          lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
          lk.inputBuffer = ib.inputBuffer;
          lr = f.nvEncLockInputBuffer(enc, &lk);
          pitch = lk.pitch;
          if (!lr) f.nvEncUnlockInputBuffer(enc, ib.inputBuffer);
          f.nvEncDestroyInputBuffer(enc, ib.inputBuffer);
        }
        std::printf("CreateInputBuffer %-9s %dx%u: %d; lock %d pitch %u\n", fm.name, w, H, r, lr, pitch);
      }
    // The pitch over a range of widths, for the formats the encoder takes.
    for (const auto& fm : {formats[0], formats[1], formats[3], formats[6]}) {
      std::printf("pitch %-7s:", fm.name);
      for (uint32_t w : {145u, 192u, 200u, 257u, 320u, 513u, 640u, 1000u, 1280u, 1920u, 3000u, 4096u}) {
        NV_ENC_CREATE_INPUT_BUFFER ib{};
        ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
        ib.width = w;
        ib.height = 128;
        poke(ib.bufferFmt, fm.fmt);
        uint32_t pitch = 0;
        if (!f.nvEncCreateInputBuffer(enc, &ib)) {
          NV_ENC_LOCK_INPUT_BUFFER lk{};
          lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
          lk.inputBuffer = ib.inputBuffer;
          if (!f.nvEncLockInputBuffer(enc, &lk)) {
            pitch = lk.pitch;
            f.nvEncUnlockInputBuffer(enc, ib.inputBuffer);
          }
          f.nvEncDestroyInputBuffer(enc, ib.inputBuffer);
        }
        std::printf(" %u:%u", w, pitch);
      }
      std::printf("\n");
    }
    {
      NV_ENC_CREATE_INPUT_BUFFER ib{};
      ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
      ib.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
      std::printf("CreateInputBuffer (0x0): %d\n", f.nvEncCreateInputBuffer(enc, &ib));
      ib.width = W;
      ib.height = H;
      ib.version = 0;
      std::printf("CreateInputBuffer (version 0): %d\n", f.nvEncCreateInputBuffer(enc, &ib));
      std::printf("CreateInputBuffer (null): %d\n", f.nvEncCreateInputBuffer(enc, nullptr));
      std::printf("DestroyInputBuffer (null buffer): %d\n", f.nvEncDestroyInputBuffer(enc, nullptr));
      std::printf("LockInputBuffer (null): %d\n", f.nvEncLockInputBuffer(enc, nullptr));
      NV_ENC_LOCK_INPUT_BUFFER lk{};
      lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
      std::printf("LockInputBuffer (null buffer): %d\n", f.nvEncLockInputBuffer(enc, &lk));
      std::printf("UnlockInputBuffer (null): %d\n", f.nvEncUnlockInputBuffer(enc, nullptr));
    }

    // One frame all the way: input, bitstream, encode, lock.
    NV_ENC_CREATE_INPUT_BUFFER ib{};
    ib.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
    ib.width = W;
    ib.height = H;
    ib.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    f.nvEncCreateInputBuffer(enc, &ib);
    NV_ENC_CREATE_BITSTREAM_BUFFER ob{};
    ob.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    std::printf("CreateBitstreamBuffer: %d\n", f.nvEncCreateBitstreamBuffer(enc, &ob));
    {
      NV_ENC_CREATE_BITSTREAM_BUFFER bad{};
      std::printf("CreateBitstreamBuffer (version 0): %d\n", f.nvEncCreateBitstreamBuffer(enc, &bad));
      std::printf("CreateBitstreamBuffer (null): %d\n", f.nvEncCreateBitstreamBuffer(enc, nullptr));
      std::printf("DestroyBitstreamBuffer (null): %d\n", f.nvEncDestroyBitstreamBuffer(enc, nullptr));
    }
    NV_ENC_LOCK_INPUT_BUFFER lk{};
    lk.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
    lk.inputBuffer = ib.inputBuffer;
    f.nvEncLockInputBuffer(enc, &lk);
    for (uint32_t y = 0; y < H * 3 / 2; ++y)
      for (uint32_t x = 0; x < W; ++x) static_cast<uint8_t*>(lk.bufferDataPtr)[y * lk.pitch + x] = static_cast<uint8_t>(16 + (x * 3 + y * 5) % 200);
    f.nvEncUnlockInputBuffer(enc, ib.inputBuffer);
    (void)lk;
    // (the input buffer is device memory here? on the card it is host-mapped; either way the pointer is writable by the CPU above)

    const auto pic = [&](NV_ENC_INPUT_PTR in, NV_ENC_OUTPUT_PTR out) {
      NV_ENC_PIC_PARAMS pp{};
      pp.version = NV_ENC_PIC_PARAMS_VER;
      pp.inputBuffer = in;
      pp.outputBitstream = out;
      pp.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
      pp.inputWidth = W;
      pp.inputHeight = H;
      pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
      pp.inputTimeStamp = 90000;
      pp.inputDuration = 3000;
      return pp;
    };
    // A picture that encodes leaves its bitstream waiting in the output buffer
    // until it is locked; the calls below that succeed are drained at once.
    const auto drain = [&](const char* what) {
      NV_ENC_LOCK_BITSTREAM lb{};
      lb.version = NV_ENC_LOCK_BITSTREAM_VER;
      lb.outputBitstream = ob.bitstreamBuffer;
      const int r = f.nvEncLockBitstream(enc, &lb);
      std::printf("  (%s) lock %d, size>0 %d\n", what, r, lb.bitstreamSizeInBytes > 0);
      f.nvEncUnlockBitstream(enc, ob.bitstreamBuffer);
    };
    NV_ENC_PIC_PARAMS pp = pic(ib.inputBuffer, ob.bitstreamBuffer);
    pp.frameIdx = 7;
    std::printf("EncodePicture: %d\n", f.nvEncEncodePicture(enc, &pp));
    {
      NV_ENC_LOCK_BITSTREAM lb{};
      lb.version = NV_ENC_LOCK_BITSTREAM_VER;
      lb.outputBitstream = ob.bitstreamBuffer;
      const int r = f.nvEncLockBitstream(enc, &lb);
      std::printf("LockBitstream: %d size>0 %d ptr %d ts %llu duration %llu frameIdx %u type %d struct %d slices %u hw %u temporal %u\n", r,
                  lb.bitstreamSizeInBytes > 0, lb.bitstreamBufferPtr != nullptr, static_cast<unsigned long long>(lb.outputTimeStamp),
                  static_cast<unsigned long long>(lb.outputDuration), lb.frameIdx, static_cast<int>(lb.pictureType),
                  static_cast<int>(lb.pictureStruct), lb.numSlices, lb.hwEncodeStatus, lb.temporalId);
      // The NAL units of the first frame.
      std::printf("first frame NAL types:");
      const uint8_t* b = static_cast<const uint8_t*>(lb.bitstreamBufferPtr);
      for (uint32_t i = 0; i + 4 < lb.bitstreamSizeInBytes; ++i)
        if (b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 0 && b[i + 3] == 1) std::printf(" %d", b[i + 4] & 31);
      std::printf("\n");
      std::printf("UnlockBitstream: %d\n", f.nvEncUnlockBitstream(enc, ob.bitstreamBuffer));
      std::printf("UnlockBitstream (unlocked): %d\n", f.nvEncUnlockBitstream(enc, ob.bitstreamBuffer));
      std::printf("UnlockBitstream (null): %d\n", f.nvEncUnlockBitstream(enc, nullptr));
      NV_ENC_LOCK_BITSTREAM bad{};
      std::printf("LockBitstream (version 0): %d\n", f.nvEncLockBitstream(enc, &bad));
      bad.version = NV_ENC_LOCK_BITSTREAM_VER;
      std::printf("LockBitstream (null buffer): %d\n", f.nvEncLockBitstream(enc, &bad));
      std::printf("LockBitstream (null): %d\n", f.nvEncLockBitstream(enc, nullptr));
    }

    {
      NV_ENC_PIC_PARAMS pp = pic(ib.inputBuffer, ob.bitstreamBuffer);
      pp.version = 0;
      std::printf("EncodePicture (version 0): %d\n", f.nvEncEncodePicture(enc, &pp));
      std::printf("EncodePicture (null): %d\n", f.nvEncEncodePicture(enc, nullptr));
      pp = pic(nullptr, ob.bitstreamBuffer);
      std::printf("EncodePicture (null input): %d\n", f.nvEncEncodePicture(enc, &pp));
      pp = pic(ib.inputBuffer, nullptr);
      std::printf("EncodePicture (null output): %d\n", f.nvEncEncodePicture(enc, &pp));
      pp = pic(ib.inputBuffer, ob.bitstreamBuffer);
      pp.inputWidth = W + 16;
      int r = f.nvEncEncodePicture(enc, &pp);
      std::printf("EncodePicture (wrong width): %d\n", r);
      if (!r) drain("wrong width");
      pp = pic(ib.inputBuffer, ob.bitstreamBuffer);
      pp.bufferFmt = NV_ENC_BUFFER_FORMAT_ARGB;
      r = f.nvEncEncodePicture(enc, &pp);
      std::printf("EncodePicture (wrong format): %d\n", r);
      if (!r) drain("wrong format");
      pp = pic(ib.inputBuffer, ob.bitstreamBuffer);
      poke(pp.pictureStruct, 9);
      std::printf("EncodePicture (bad picture structure): %d\n", f.nvEncEncodePicture(enc, &pp));
    }
    // Sequence parameters: SPS then PPS.
    {
      uint8_t buf[512];
      uint32_t payload = 0;
      NV_ENC_SEQUENCE_PARAM_PAYLOAD sp{};
      sp.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
      sp.inBufferSize = sizeof buf;
      sp.spsppsBuffer = buf;
      sp.outSPSPPSPayloadSize = &payload;
      const int r = f.nvEncGetSequenceParams(enc, &sp);
      std::printf("GetSequenceParams: %d, NAL types:", r);
      for (uint32_t i = 0; i + 4 < payload; ++i)
        if (buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 0 && buf[i + 3] == 1) std::printf(" %d", buf[i + 4] & 31);
      std::printf("\n");
      sp.inBufferSize = 8;
      std::printf("GetSequenceParams (8 bytes of room): %d\n", f.nvEncGetSequenceParams(enc, &sp));
      sp.inBufferSize = sizeof buf;
      sp.version = 0;
      std::printf("GetSequenceParams (version 0): %d\n", f.nvEncGetSequenceParams(enc, &sp));
      sp.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
      sp.spsppsBuffer = nullptr;
      std::printf("GetSequenceParams (null buffer): %d\n", f.nvEncGetSequenceParams(enc, &sp));
    }
    // Statistics.
    {
      NV_ENC_STAT st{};
      st.version = NV_ENC_STAT_VER;
      st.outputBitStream = ob.bitstreamBuffer;
      std::printf("GetEncodeStats: %d\n", f.nvEncGetEncodeStats(enc, &st));
    }
    // Resources registered from device memory and mapped.
    {
      void* dev = nullptr;
      size_t pitch = 0;
      cudaMallocPitch(&dev, &pitch, W, H * 3 / 2);
      cudaMemset2D(dev, pitch, 90, W, H * 3 / 2);
      NV_ENC_REGISTER_RESOURCE rr{};
      rr.version = NV_ENC_REGISTER_RESOURCE_VER;
      rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
      rr.width = W;
      rr.height = H;
      rr.pitch = static_cast<uint32_t>(pitch);
      rr.resourceToRegister = dev;
      rr.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
      rr.bufferUsage = NV_ENC_INPUT_IMAGE;
      std::printf("RegisterResource: %d\n", f.nvEncRegisterResource(enc, &rr));
      NV_ENC_MAP_INPUT_RESOURCE mp{};
      mp.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
      mp.registeredResource = rr.registeredResource;
      const int r = f.nvEncMapInputResource(enc, &mp);
      std::printf("MapInputResource: %d, format 0x%x\n", r, mp.mappedBufferFmt);
      NV_ENC_PIC_PARAMS mpp = pic(mp.mappedResource, ob.bitstreamBuffer);
      std::printf("EncodePicture (mapped resource): %d\n", f.nvEncEncodePicture(enc, &mpp));
      NV_ENC_LOCK_BITSTREAM lb{};
      lb.version = NV_ENC_LOCK_BITSTREAM_VER;
      lb.outputBitstream = ob.bitstreamBuffer;
      std::printf("LockBitstream: %d\n", f.nvEncLockBitstream(enc, &lb));
      f.nvEncUnlockBitstream(enc, ob.bitstreamBuffer);
      std::printf("MapInputResource (again): %d\n", f.nvEncMapInputResource(enc, &mp));
      std::printf("UnmapInputResource: %d\n", f.nvEncUnmapInputResource(enc, mp.mappedResource));
      std::printf("UnmapInputResource (again): %d\n", f.nvEncUnmapInputResource(enc, mp.mappedResource));
      std::printf("UnregisterResource: %d\n", f.nvEncUnregisterResource(enc, rr.registeredResource));
      NV_ENC_REGISTER_RESOURCE bad = rr;
      bad.registeredResource = nullptr;
      bad.resourceToRegister = nullptr;
      std::printf("RegisterResource (null pointer): %d\n", f.nvEncRegisterResource(enc, &bad));
      bad = rr;
      bad.version = 0;
      std::printf("RegisterResource (version 0): %d\n", f.nvEncRegisterResource(enc, &bad));
      bad = rr;
      bad.bufferFormat = NV_ENC_BUFFER_FORMAT_UNDEFINED;
      std::printf("RegisterResource (undefined format): %d\n", f.nvEncRegisterResource(enc, &bad));
      bad = rr;
      bad.width = 0;
      std::printf("RegisterResource (zero width): %d\n", f.nvEncRegisterResource(enc, &bad));
      std::printf("UnregisterResource (null): %d\n", f.nvEncUnregisterResource(enc, nullptr));
      std::printf("MapInputResource (null): %d\n", f.nvEncMapInputResource(enc, nullptr));
      cudaFree(dev);
    }
    // The rest of the table.
    {
      NV_ENC_RECONFIGURE_PARAMS rc{};
      rc.version = NV_ENC_RECONFIGURE_PARAMS_VER;
      rc.reInitEncodeParams = ip;
      std::printf("ReconfigureEncoder (same): %d\n", f.nvEncReconfigureEncoder(enc, &rc));
      rc.reInitEncodeParams.encodeWidth = 256;
      rc.reInitEncodeParams.encodeHeight = 160;
      std::printf("ReconfigureEncoder (larger): %d\n", f.nvEncReconfigureEncoder(enc, &rc));
      last_error("reconfigure to a larger size");
      rc.reInitEncodeParams = ip;
      rc.reInitEncodeParams.encodeWidth = 160;
      rc.reInitEncodeParams.encodeHeight = 96;
      std::printf("ReconfigureEncoder (smaller): %d\n", f.nvEncReconfigureEncoder(enc, &rc));
      rc.version = 0;
      std::printf("ReconfigureEncoder (version 0): %d\n", f.nvEncReconfigureEncoder(enc, &rc));
      std::printf("ReconfigureEncoder (null): %d\n", f.nvEncReconfigureEncoder(enc, nullptr));
      std::printf("InvalidateRefFrames: %d\n", f.nvEncInvalidateRefFrames(enc, 0));
      NV_ENC_EVENT_PARAMS ev{};
      ev.version = NV_ENC_EVENT_PARAMS_VER;
      std::printf("RegisterAsyncEvent: %d\n", f.nvEncRegisterAsyncEvent(enc, &ev));
      std::printf("UnregisterAsyncEvent: %d\n", f.nvEncUnregisterAsyncEvent(enc, &ev));
      cudaStream_t s1;
      cudaStreamCreate(&s1);
      std::printf("SetIOCudaStreams: %d\n", f.nvEncSetIOCudaStreams(enc, &s1, &s1));
      cudaStreamDestroy(s1);
      NV_ENC_CREATE_MV_BUFFER mv{};
      mv.version = NV_ENC_CREATE_MV_BUFFER_VER;
      std::printf("CreateMVBuffer: %d\n", f.nvEncCreateMVBuffer(enc, &mv));
      std::printf("DestroyMVBuffer: %d\n", f.nvEncDestroyMVBuffer(enc, mv.mvBuffer));
      NV_ENC_MEONLY_PARAMS me{};
      me.version = NV_ENC_MEONLY_PARAMS_VER;
      std::printf("RunMotionEstimationOnly: %d\n", f.nvEncRunMotionEstimationOnly(enc, &me));
      NV_ENC_LOOKAHEAD_PIC_PARAMS la{};
      la.version = NV_ENC_LOOKAHEAD_PIC_PARAMS_VER;
      std::printf("LookaheadPicture: %d\n", f.nvEncLookaheadPicture(enc, &la));
      NV_ENC_RESTORE_ENCODER_STATE_PARAMS rs{};
      rs.version = NV_ENC_RESTORE_ENCODER_STATE_PARAMS_VER;
      std::printf("RestoreEncoderState: %d\n", f.nvEncRestoreEncoderState(enc, &rs));
      uint8_t buf[512];
      uint32_t payload = 0;
      NV_ENC_SEQUENCE_PARAM_PAYLOAD sp{};
      sp.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
      sp.inBufferSize = sizeof buf;
      sp.spsppsBuffer = buf;
      sp.outSPSPPSPayloadSize = &payload;
      std::printf("GetSequenceParamEx: %d\n", f.nvEncGetSequenceParamEx(enc, &ip, &sp));
    }
    // End of stream.
    {
      NV_ENC_PIC_PARAMS eos{};
      eos.version = NV_ENC_PIC_PARAMS_VER;
      eos.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
      std::printf("EncodePicture (end of stream): %d\n", f.nvEncEncodePicture(enc, &eos));
    }
    f.nvEncDestroyBitstreamBuffer(enc, ob.bitstreamBuffer);
    std::printf("DestroyInputBuffer: %d\n", f.nvEncDestroyInputBuffer(enc, ib.inputBuffer));
    std::printf("DestroyEncoder: %d\n", f.nvEncDestroyEncoder(enc));
    enc = nullptr;
  }
  std::printf("DONE\n");
  return 0;
}
