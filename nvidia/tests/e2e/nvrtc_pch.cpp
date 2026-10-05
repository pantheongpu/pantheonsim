// NVRTC's precompiled-header heap and status calls, the flow callback, and
// the Tile IR getters (NVRTC 12.8 and later). The entry points are looked up
// by name, because older headers do not declare them; a library without them
// skips the program. The expectations are what NVRTC 13.2 answered (no GPU is
// involved), and the program passes against it:
//  - the PCH heap rounds a request up to whole pages;
//  - a program that asked for no PCH reports "no PCH create attempted";
//  - a flow callback is called with its payload, and a callback that answers
//    1 cancels the compile (NVRTC_ERROR_CANCELLED, 16);
//  - a program that made no Tile IR has a Tile IR size of 0, and the getter writes nothing.
#include <nvrtc.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <string>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

typedef int (*GetHeapFn)(size_t*);
typedef int (*SetHeapFn)(size_t);
typedef int (*StatusFn)(nvrtcProgram);
typedef int (*RequiredFn)(nvrtcProgram, size_t*);
typedef int (*FlowFn)(nvrtcProgram, int (*)(void*, void*), void*);
typedef int (*TileSizeFn)(nvrtcProgram, size_t*);
typedef int (*TileFn)(nvrtcProgram, char*);

static int g_calls = 0, g_answer = 0;
static void* g_payload_seen = nullptr;
static int callback(void* payload, void* reserved) {
  ++g_calls;
  g_payload_seen = payload;
  return reserved == nullptr ? g_answer : -1;
}

static nvrtcProgram make() {
  nvrtcProgram p = nullptr;
  nvrtcCreateProgram(&p, "extern \"C\" __global__ void k(float* x) { x[0] = 1.0f; }", "k.cu", 0, nullptr, nullptr);
  return p;
}

int main() {
  auto get_heap = (GetHeapFn)dlsym(RTLD_DEFAULT, "nvrtcGetPCHHeapSize");
  auto set_heap = (SetHeapFn)dlsym(RTLD_DEFAULT, "nvrtcSetPCHHeapSize");
  auto status = (StatusFn)dlsym(RTLD_DEFAULT, "nvrtcGetPCHCreateStatus");
  auto required = (RequiredFn)dlsym(RTLD_DEFAULT, "nvrtcGetPCHHeapSizeRequired");
  auto flow = (FlowFn)dlsym(RTLD_DEFAULT, "nvrtcSetFlowCallback");
  auto tile_size = (TileSizeFn)dlsym(RTLD_DEFAULT, "nvrtcGetTileIRSize");
  auto tile = (TileFn)dlsym(RTLD_DEFAULT, "nvrtcGetTileIR");
  if (!get_heap || !set_heap || !status || !required || !flow) {
    std::printf("SKIP: this NVRTC has no PCH or flow-callback entry points\n");
    return 0;
  }
  const int kInvalidInput = 3, kInvalidProgram = 4, kNoPch = 13, kCancelled = 16;

  // The heap.
  size_t was = 0;
  check(get_heap(&was) == NVRTC_SUCCESS && was > 0, "the PCH heap has a default size");
  check(get_heap(nullptr) == kInvalidInput, "a NULL heap-size result is refused");
  size_t got = 0;
  check(set_heap(5000) == NVRTC_SUCCESS && get_heap(&got) == NVRTC_SUCCESS && got >= 5000 && got % 4096 == 0,
        "a heap request rounds up to whole pages");
  check(set_heap(8 << 20) == NVRTC_SUCCESS && get_heap(&got) == NVRTC_SUCCESS && got == (size_t)(8 << 20),
        "a page-multiple request is kept as asked");
  set_heap(was);

  // Status before and after a plain compile.
  nvrtcProgram p = make();
  size_t need = 77;
  check(status(p) == kNoPch, "no PCH was attempted before any compile");
  check(required(p, &need) == NVRTC_SUCCESS && need == 0, "no heap is required before any compile");
  check(status(nullptr) == kInvalidProgram, "the status of no program");
  check(required(nullptr, &need) == kInvalidProgram, "the heap required by no program");
  check(required(p, nullptr) == kInvalidInput, "a NULL required-size result is refused");
  const char* opts[] = {"--gpu-architecture=compute_80"};
  check(nvrtcCompileProgram(p, 1, opts) == NVRTC_SUCCESS, "compile");
  check(status(p) == kNoPch, "a compile that asked for no PCH attempted none");
  need = 77;
  check(required(p, &need) == NVRTC_SUCCESS && need == 0, "...and requires no heap");

  // Tile IR: none was made.
  if (tile_size && tile) {
    size_t t = 99;
    char c = 'x';
    check(tile_size(p, &t) == NVRTC_SUCCESS && t == 0, "the Tile IR size is 0");
    check(tile(p, &c) == NVRTC_SUCCESS && c == 'x', "the Tile IR getter succeeds and writes nothing");
    check(tile_size(p, nullptr) == kInvalidInput && tile(p, nullptr) == kInvalidInput, "NULL Tile IR outputs");
    check(tile_size(nullptr, &t) == kInvalidProgram && tile(nullptr, &c) == kInvalidProgram, "Tile IR of no program");
  }
  nvrtcDestroyProgram(&p);

  // The flow callback.
  p = make();
  int payload = 0;
  check(flow(nullptr, callback, &payload) == kInvalidProgram, "a flow callback on no program");
  check(flow(p, nullptr, &payload) == kInvalidInput, "a NULL flow callback is refused");
  check(flow(p, callback, &payload) == NVRTC_SUCCESS, "set the flow callback");
  g_calls = 0;
  g_answer = 0;
  check(nvrtcCompileProgram(p, 1, opts) == NVRTC_SUCCESS, "a callback that answers 0 lets the compile finish");
  check(g_calls >= 1 && g_payload_seen == &payload, "the callback was called with its payload");
  nvrtcDestroyProgram(&p);

  p = make();
  flow(p, callback, &payload);
  g_calls = 0;
  g_answer = 1;
  const int rc = nvrtcCompileProgram(p, 1, opts);
  check(rc == kCancelled, "a callback that answers 1 cancels the compile");
  check(std::strcmp(nvrtcGetErrorString((nvrtcResult)rc), "NVRTC_ERROR_CANCELLED") == 0, "the cancellation has a name");
  nvrtcDestroyProgram(&p);

  std::printf(failures ? "FAIL: %d NVRTC PCH checks\n" : "PASS: every NVRTC PCH check\n", failures);
  return failures ? 1 : 0;
}
