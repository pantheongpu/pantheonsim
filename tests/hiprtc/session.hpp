// One session of hiprtc calls as a program makes them, printed byte for byte (as sizes and hashes): what
// tests/hiprtc/test_hiprtc_cache.cpp runs in its child processes against the stand-in library, and tests/hiprtc/hiprtc_probe.cpp
// against ROCm's own. The prototypes are hiprtc.h's (ROCm's headers are not needed to build this).
#pragma once
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

extern "C" {
const char* hiprtcGetErrorString(int);
int hiprtcVersion(int*, int*);
int hiprtcCreateProgram(void**, const char*, const char*, int, const char**, const char**);
int hiprtcDestroyProgram(void**);
int hiprtcAddNameExpression(void*, const char*);
int hiprtcCompileProgram(void*, int, const char**);
int hiprtcGetCodeSize(void*, size_t*);
int hiprtcGetCode(void*, char*);
int hiprtcGetProgramLogSize(void*, size_t*);
int hiprtcGetProgramLog(void*, char*);
int hiprtcGetLoweredName(void*, const char*, const char**);
int hiprtcGetBitcodeSize(void*, size_t*);
int hiprtcGetBitcode(void*, char*);
int hiprtcLinkCreate(unsigned, void*, void**, void**);
int hiprtcLinkAddData(void*, int, void*, size_t, const char*, unsigned, void*, void**);
int hiprtcLinkComplete(void*, void**, size_t*);
int hiprtcLinkDestroy(void*);
}

namespace hiprtc_session {

inline uint64_t fnv(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (const unsigned char c : s) h = (h ^ c) * 1099511628211ull;
  return h;
}

// A program: `src` with `name`, headers ("file=text"), options, name expressions; `how` is the order of calls.
struct Spec {
  std::string src = "__global__ void k(float* a) { a[0] = 1.0f; }", name = "k.cu", how = "plain";
  std::vector<std::string> headers, options, exprs;
};
inline std::string hexbytes(const std::string& s) {
  char b[64];
  std::snprintf(b, sizeof b, "%zu:%016llx", s.size(), static_cast<unsigned long long>(fnv(s)));
  return b;
}
inline std::string session(const Spec& s) {
  std::string out;
  const auto say = [&](const std::string& what, int r) { out += what + "=" + std::to_string(r) + "(" + hiprtcGetErrorString(r) + ") "; };
  std::vector<std::string> names, texts;
  for (const std::string& h : s.headers) names.push_back(h.substr(0, h.find('='))), texts.push_back(h.substr(h.find('=') + 1));
  std::vector<const char*> pn, pt, po;
  for (size_t i = 0; i < names.size(); ++i) pn.push_back(names[i].c_str()), pt.push_back(texts[i].c_str());
  for (const std::string& o : s.options) po.push_back(o.c_str());
  void* p = nullptr;
  say("create", hiprtcCreateProgram(&p, s.src.c_str(), s.name.c_str(), static_cast<int>(pn.size()), pt.empty() ? nullptr : pt.data(),
                                    pn.empty() ? nullptr : pn.data()));
  if (s.how == "early") {   // asked for before any compile
    size_t n = 77;
    say("early-codesize", hiprtcGetCodeSize(p, &n));
  }
  for (const std::string& e : s.exprs) say("addname", hiprtcAddNameExpression(p, e.c_str()));
  const int c = hiprtcCompileProgram(p, static_cast<int>(po.size()), po.empty() ? nullptr : po.data());
  say("compile", c);
  if (s.how == "twice") say("compile2", hiprtcCompileProgram(p, static_cast<int>(po.size()), po.empty() ? nullptr : po.data()));
  if (s.how == "late-expr") say("late-addname", hiprtcAddNameExpression(p, "late"));
  size_t log_size = 99;
  say("logsize", hiprtcGetProgramLogSize(p, &log_size));
  std::string log(log_size + 32, '#');
  say("log", hiprtcGetProgramLog(p, log.data()));
  // (ROCm's compiler names its temporary directory, /tmp/comgr-<six hex digits>, in every diagnostic: a different one each compile)
  for (size_t at = log.find("comgr-"); at != std::string::npos; at = log.find("comgr-", at + 6))
    for (size_t k = at + 6; k < at + 12 && k < log.size() && std::isxdigit(static_cast<unsigned char>(log[k])); ++k) log[k] = 'X';
  out += "log:" + std::to_string(log_size) + ":" + hexbytes(log) + " ";
  if (c == 0) {
    size_t code_size = 0;
    say("codesize", hiprtcGetCodeSize(p, &code_size));
    std::string code(code_size + 32, '#');
    say("code", hiprtcGetCode(p, code.data()));
    out += "code:" + std::to_string(code_size) + ":" + hexbytes(code) + " ";
    for (const std::string& e : s.exprs) {
      const char* lowered = nullptr;
      say("lowered", hiprtcGetLoweredName(p, e.c_str(), &lowered));
      out += std::string("lowered:") + (lowered ? lowered : "(null)") + " ";
    }
    if (s.how == "unknown-name") {
      const char* lowered = nullptr;
      say("unknown", hiprtcGetLoweredName(p, "never added", &lowered));
    }
    if (s.how == "bitcode") {
      size_t n = 0;
      say("bitcodesize", hiprtcGetBitcodeSize(p, &n));
      std::string bc(n + 8, '#');
      say("bitcode", hiprtcGetBitcode(p, bc.data()));
      out += "bitcode:" + hexbytes(bc) + " ";
      // ... and the code again, after the real program was made
      std::string again(code_size + 32, '#');
      say("code-again", hiprtcGetCode(p, again.data()));
      out += "code-again:" + hexbytes(again) + " ";
    }
    if (s.how == "null-args") {
      say("codesize-null", hiprtcGetCodeSize(p, nullptr));
      say("lowered-null", hiprtcGetLoweredName(p, nullptr, nullptr));
    }
  }
  say("destroy", hiprtcDestroyProgram(&p));
  out += p ? "handle-left " : "handle-cleared ";
  return out;
}

// The arguments of a child: --src S --name N --header F=T --opt O --expr E --how H; `--next` starts another program in the same
// process; --repeat N runs the whole list N times. Prints one line a session.
inline int child(int argc, char** argv) {
  std::vector<Spec> specs(1);
  int repeat = 1;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--src") specs.back().src = value();
    else if (a == "--src-file") {
      std::ifstream in(value(), std::ios::binary);
      specs.back().src.assign((std::istreambuf_iterator<char>(in)), {});
    }
    else if (a == "--name") specs.back().name = value();
    else if (a == "--header") specs.back().headers.push_back(value());
    else if (a == "--opt") specs.back().options.push_back(value());
    else if (a == "--expr") specs.back().exprs.push_back(value());
    else if (a == "--how") specs.back().how = value();
    else if (a == "--next") specs.emplace_back();
    else if (a == "--repeat") repeat = std::atoi(value().c_str());
  }
  if (argc > 2 && !std::strcmp(argv[2], "--misc")) {   // the calls that are not about a program
    int major = 0, minor = 0;
    const int v = hiprtcVersion(&major, &minor);
    std::printf("version=%d %d.%d ", v, major, minor);
    void* state = nullptr;
    std::printf("linkcreate=%d ", hiprtcLinkCreate(0, nullptr, nullptr, &state));
    char image[] = "BITCODE";
    std::printf("linkadd=%d ", hiprtcLinkAddData(state, 0, image, 7, "m", 0, nullptr, nullptr));
    void* bin = nullptr;
    size_t size = 0;
    const int done = hiprtcLinkComplete(state, &bin, &size);
    std::printf("linkcomplete=%d %s ", done, bin ? std::string(static_cast<char*>(bin), size).c_str() : "(null)");
    std::printf("linkdestroy=%d ", hiprtcLinkDestroy(state));
    void* none = nullptr;
    std::printf("destroy-null=%d compile-null=%d\n", hiprtcDestroyProgram(&none), hiprtcCompileProgram(nullptr, 0, nullptr));
    return 0;
  }
  for (int r = 0; r < repeat; ++r)
    for (const Spec& s : specs) std::printf("%s\n", session(s).c_str());
  return 0;
}

}  // namespace hiprtc_session
