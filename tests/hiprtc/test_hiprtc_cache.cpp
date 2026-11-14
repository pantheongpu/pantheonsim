// The simulator's libhiprtc (amd/src/hiprtc_cache.cpp) against a stand-in for ROCm's (tests/hiprtc/fake_hiprtc.cpp), without ROCm.
//
//   test_hiprtc_cache <directory with libfake_hiprtc_real.so, libfake_comgr_a.so, libfake_comgr_b.so, libfake_builtins_a.so, libfake_builtins_b.so>
//
// The test makes an installation in a temporary directory, shaped as ROCm's is (lib/libhiprtc.so.6 with the compiler library
// and the built-in header library beside it, llvm/bin/rocm.cfg, llvm/lib/clang/19/lib/amdgcn/bitcode/hip.bc,
// include/hip/hip_runtime.h), and runs itself as a child process once for each compile session: the child calls hiprtc as a
// program does and prints what it got back, byte for byte (as hashes), and the stand-in counts the compiles it really did in a
// file. What is held:
//   * what a program gets back with the cache is what it gets with the cache off (VGPU_HIPRTC_CACHE=0): result codes, code, log
//     size and bytes, lowered names, bitcode -- from a cold cache, from memory, from a file in a later process;
//   * a compile is done once for the same inputs, in a process and across processes, and again when anything it depends on
//     changes: source, name, a header or its name, an option, a name expression, the device's architecture, the environment,
//     the compiler library, the built-in header, the bitcode and configuration files;
//   * what cannot be named completely is compiled every time (an include that was not given, __DATE__, -g, -include ...), and
//     <hip/hip_runtime.h> is cacheable exactly when it is the guarded header and nothing can redirect it;
//   * a failed compile is not kept; calls the cache has no answer for (the bitcode, a second compile, a late name expression) go to
//     the real library and give its answers; the link API is forwarded;
//   * a damaged, truncated, empty, foreign or misnamed entry is not used and is replaced; eight processes filling a cold cache at
//     once all get the right bytes and leave whole entries and no temporary files; the size cap holds; a directory that is not
//     the user's alone is not used; `mem` and `0` write nothing; the statistics line is printed when asked for;
//   * every entry point the real library exports is defined, and a missing real library is an error code, not a crash.
// Framework-free: prints what failed and exits non-zero. The program exports hipGetDevice and hipGetDevicePropertiesR0600
// (TEST_ARCH names the architecture), which is how the cache and the stand-in learn the device.
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "session.hpp"

extern "C" {
// The HIP runtime's two calls hiprtc makes, standing in for the simulator's: one device, whose architecture is $TEST_ARCH.
__attribute__((visibility("default"))) int hipGetDevice(int* id) {
  if (std::getenv("TEST_NO_DEVICE")) return 1;
  *id = 0;
  return 0;
}
__attribute__((visibility("default"))) int hipGetDevicePropertiesR0600(void* props, int) {
  char* p = static_cast<char*>(props);
  std::snprintf(p, 256, "Fake Instinct");
  const char* arch = std::getenv("TEST_ARCH");
  std::snprintf(p + 1024, 256, "%s", arch ? arch : "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-");
  return 0;
}
}

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  if (!ok) ++failures;
  std::printf("%s  %s\n", ok ? "ok   " : "FAIL ", what.c_str());
  std::fflush(stdout);
}

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), {});
}
void spit(const std::string& path, const std::string& bytes) { std::ofstream(path, std::ios::binary) << bytes; }
void mkdirs(const std::string& path) {
  for (size_t at = 1; at <= path.size(); ++at)
    if (at == path.size() || path[at] == '/') ::mkdir(path.substr(0, at).c_str(), 0755);
}
std::vector<std::string> files_in(const std::string& dir) {
  std::vector<std::string> out;
  if (DIR* d = ::opendir(dir.c_str())) {
    while (const dirent* e = ::readdir(d))
      if (std::strcmp(e->d_name, ".") && std::strcmp(e->d_name, "..")) out.push_back(e->d_name);
    ::closedir(d);
  }
  return out;
}
uint64_t bytes_in(const std::string& dir) {
  uint64_t n = 0;
  for (const std::string& f : files_in(dir)) {
    struct stat st;
    if (::stat((dir + "/" + f).c_str(), &st) == 0) n += static_cast<uint64_t>(st.st_size);
  }
  return n;
}

// ---- the parent

std::string g_self, g_root, g_cache, g_log;
std::string sh_quote(const std::string& s) {
  std::string q = "'";
  for (const char c : s) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return q + "'";
}
// Runs a child with `env` ("A=b C=d", put before the command) and `args`; returns its standard output (and its standard error where
// `with_stderr`).
std::string run(const std::string& env, const std::vector<std::string>& args, bool with_stderr = false) {
  std::string cmd = "env VGPU_HIPRTC_REAL=" + sh_quote(g_root + "/lib/libhiprtc.so.6") + " VGPU_HIPRTC_CACHE_DIR=" + sh_quote(g_cache) + " FAKE_HIPRTC_LOG=" +
                    sh_quote(g_log) + " " + env + " " + sh_quote(g_self) + " child";
  for (const std::string& a : args) cmd += " " + sh_quote(a);
  cmd += with_stderr ? " 2>&1" : " 2>/dev/null";
  std::string out;
  if (FILE* f = ::popen(cmd.c_str(), "r")) {
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    ::pclose(f);
  }
  return out;
}
size_t compiles() {
  const std::string log = slurp(g_log);
  size_t n = 0;
  for (const char c : log) n += c == '\n';
  return n;
}
void reset_cache() {
  for (const std::string& f : files_in(g_cache)) ::unlink((g_cache + "/" + f).c_str());
  ::chmod(g_cache.c_str(), 0700);
}
// How many compiles the stand-in did for `args` under `env`, and the output, which must be what the cache-off run gives.
struct Outcome {
  size_t compiles;
  std::string out;
};
Outcome measure(const std::string& env, const std::vector<std::string>& args) {
  const size_t before = compiles();
  const std::string out = run(env, args);
  return {compiles() - before, out};
}
std::string off(const std::string& env, const std::vector<std::string>& args) { return run("VGPU_HIPRTC_CACHE=0 " + env, args); }

void copy(const std::string& from, const std::string& to) {
  ::unlink(to.c_str());   // (a new file, not new bytes in the old one: a process that has the old one mapped keeps it)
  spit(to, slurp(from));
  ::chmod(to.c_str(), 0755);
}

// A set of programs that covers what a program can pass: headers, options, name expressions, a warning, bytes of every kind.
std::vector<std::string> suite() {
  return {"--src", "__global__ void a(float* x) { x[0] = 1; }",
          "--next", "--src", "__global__ void b(float* x) { x[0] = 2; } // WARN", "--name", "b.hip", "--opt", "-O2", "--opt", "-DN=4", "--opt", "-ffp-contract=off",
          "--next", "--src", "#include \"mine.h\"\ntemplate <typename T> __global__ void c(T* x) { x[0] = MINE; }", "--header", "mine.h=#define MINE 3", "--expr",
          "c<float>", "--expr", "c<double>",
          "--next", "--src", "__global__ void d(float* x) { x[0] = 4; } // SIZE=70000", "--opt", "--offload-arch=gfx90a",
          "--next", "--src", "#include <hip/hip_runtime.h>\n__global__ void e(float* x) { x[0] = 5; }", "--name", "VkFFT.hip"};
}
constexpr size_t kSuite = 5;

int parent(const std::string& libs) {
  char tmpl[] = "/tmp/vgpu-hiprtc-test-XXXXXX";
  if (!::mkdtemp(tmpl)) {
    std::printf("SKIP: no temporary directory\n");
    return 0;
  }
  const std::string top = tmpl;
  g_root = top + "/rocm", g_cache = top + "/cache", g_log = top + "/compiles.log";
  mkdirs(g_root + "/lib");
  mkdirs(g_root + "/llvm/bin");
  mkdirs(g_root + "/llvm/lib/clang/19/lib/amdgcn/bitcode");
  mkdirs(g_root + "/include/hip");
  ::mkdir(g_cache.c_str(), 0700);
  copy(libs + "/libfake_hiprtc_real.so", g_root + "/lib/libhiprtc.so.6");
  copy(libs + "/libfake_comgr_a.so", g_root + "/lib/libamd_comgr.so.3");
  copy(libs + "/libfake_builtins_a.so", g_root + "/lib/libhiprtc-builtins.so");
  spit(g_root + "/llvm/bin/rocm.cfg", "-O3\n");
  spit(g_root + "/llvm/lib/clang/19/lib/amdgcn/bitcode/hip.bc", "device library 1");
  const std::string guarded = "/* ROCm's */\n#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_H\n#define HIP_INCLUDE_HIP_HIP_RUNTIME_H\n#if 1\n#include <stdint.h>\n#endif\n#endif // guard\n";
  spit(g_root + "/include/hip/hip_runtime.h", guarded);
  spit(g_log, "");
  const std::string kept = slurp(g_log);
  (void)kept;

  // ---- the same bytes with the cache off, cold, from memory and from a file
  const std::vector<std::string> all = suite();
  const std::string want = off("", all);
  check(want.find("compile=0(FAKE_SUCCESS)") != std::string::npos && want.find("code:70000:") != std::string::npos, "the stand-in compiles (cache off)");
  check(want.find("handle-left") != std::string::npos && want.find("handle-cleared") == std::string::npos,
        "(a destroyed program's handle is left as it was, as ROCm's library leaves it: the comparisons below hold the cache to that)");
  {
    const Outcome o = measure("VGPU_HIPRTC_CACHE=0", all);
    check(o.compiles == kSuite, "with the cache off every program is compiled by the real library");
    check(files_in(g_cache).empty(), "... and nothing is written");
    std::vector<std::string> twice = all;
    twice.push_back("--repeat"), twice.push_back("2");
    const Outcome again = measure("VGPU_HIPRTC_CACHE=0", twice);
    check(again.compiles == 2 * kSuite && again.out == want + want, "... nor kept in memory: a program compiled twice in a process is compiled twice");
  }
  {
    const Outcome cold = measure("", all);
    check(cold.out == want, "a cold cache hands back what the real library does, byte for byte");
    check(cold.compiles == kSuite, "... having compiled each program once");
    check(files_in(g_cache).size() == kSuite, "... and kept an entry for each");
    const Outcome warm = measure("", all);
    check(warm.out == want, "a later process gets the same bytes from the files");
    check(warm.compiles == 0, "... without one real compile");
    std::vector<std::string> thrice = all;
    thrice.push_back("--repeat"), thrice.push_back("3");
    reset_cache();
    const Outcome mem = measure("VGPU_HIPRTC_CACHE=mem", thrice);
    check(mem.out == want + want + want, "in one process the second and third compiles of a program get the first one's bytes");
    check(mem.compiles == kSuite, "... from memory: one real compile a program");
    check(files_in(g_cache).empty(), "VGPU_HIPRTC_CACHE=mem writes no file");
    const Outcome again = measure("VGPU_HIPRTC_CACHE=mem", all);
    check(again.compiles == kSuite, "... so the next process compiles again");
  }

  // ---- anything a compile depends on changes: a new compile, with that compile's bytes
  reset_cache();
  const std::vector<std::string> base = {"--src", "__global__ void k(float* x) { x[0] = 1; }", "--name", "k.cu", "--opt", "-O2", "--header", "h.h=#define H 1",
                                         "--expr", "k"};
  measure("", base);
  const auto differs = [&](const std::string& what, const std::string& env, std::vector<std::string> args) {
    const std::string w = off(env, args);
    const Outcome first = measure(env, args), second = measure(env, args);
    check(first.compiles == 1 && first.out == w && second.compiles == 0 && second.out == w, "a new compile for " + what + ", kept in its turn");
  };
  check(measure("", base).compiles == 0, "(the base program is in the cache)");
  differs("another source", "", {"--src", "__global__ void k(float* x) { x[0] = 2; }", "--name", "k.cu", "--opt", "-O2", "--header", "h.h=#define H 1", "--expr", "k"});
  differs("another program name", "", {"--src", "__global__ void k(float* x) { x[0] = 1; }", "--name", "j.cu", "--opt", "-O2", "--header", "h.h=#define H 1", "--expr", "k"});
  differs("another option", "", {"--src", "__global__ void k(float* x) { x[0] = 1; }", "--name", "k.cu", "--opt", "-O3", "--header", "h.h=#define H 1", "--expr", "k"});
  differs("one option more", "", {"--src", "__global__ void k(float* x) { x[0] = 1; }", "--name", "k.cu", "--opt", "-O2", "--opt", "-w", "--header", "h.h=#define H 1", "--expr", "k"});
  differs("another header text", "", {"--src", "__global__ void k(float* x) { x[0] = 1; }", "--name", "k.cu", "--opt", "-O2", "--header", "h.h=#define H 2", "--expr", "k"});
  differs("another header name", "", {"--src", "__global__ void k(float* x) { x[0] = 1; }", "--name", "k.cu", "--opt", "-O2", "--header", "g.h=#define H 1", "--expr", "k"});
  differs("another name expression", "", {"--src", "__global__ void k(float* x) { x[0] = 1; }", "--name", "k.cu", "--opt", "-O2", "--header", "h.h=#define H 1", "--expr", "kk"});
  differs("no name expression", "", {"--src", "__global__ void k(float* x) { x[0] = 1; }", "--name", "k.cu", "--opt", "-O2", "--header", "h.h=#define H 1"});
  differs("another device architecture", "TEST_ARCH=amdgcn-amd-amdhsa--gfx1100", base);
  differs("a compiler variable in the environment", "HIP_FAKE_FLAVOUR=7", base);
  check(measure("UNRELATED_VARIABLE=1", base).compiles == 0, "a variable the compiler does not read changes nothing");
  // (the toolchain: each change is left in place for the ones after it)
  spit(g_root + "/llvm/lib/clang/19/lib/amdgcn/bitcode/hip.bc", "device library 2, longer");
  differs("another device bitcode library", "", base);
  spit(g_root + "/llvm/bin/rocm.cfg", "-O3 -ffast-math\n");
  differs("another clang configuration file", "", base);
  copy(libs + "/libfake_comgr_b.so", g_root + "/lib/libamd_comgr.so.3");
  differs("another compiler library", "", base);
  {
    const std::string a = slurp(libs + "/libfake_builtins_a.so");
    spit(g_root + "/lib/libhiprtc-builtins.so", a + std::string(64, '\0'));   // (the same header in a file that is not the same)
    differs("another built-in header library", "", base);
    copy(libs + "/libfake_builtins_a.so", g_root + "/lib/libhiprtc-builtins.so");
  }
  {
    const Outcome o = measure("TEST_NO_DEVICE=1", base), o2 = measure("TEST_NO_DEVICE=1", base);
    check(o.compiles == 1 && o2.compiles == 1 && o.out == off("TEST_NO_DEVICE=1", base), "no device to ask for the architecture: compiled every time");
  }

  // ---- what cannot be named completely is the real library's every time
  const auto every_time = [&](const std::string& what, const std::string& env, const std::vector<std::string>& args) {
    const Outcome a = measure(env, args), b = measure(env, args);
    check(a.compiles == 1 && b.compiles == 1, what + ": compiled every time");
  };
  const auto once = [&](const std::string& what, const std::string& env, const std::vector<std::string>& args) {
    const std::string w = off(env, args);
    const Outcome a = measure(env, args), b = measure(env, args);
    check(a.compiles + b.compiles == 1 && a.out == w && b.out == w, what + ": compiled once");
  };
  reset_cache();
  spit("local.h", "#define LOCAL 1\n");
  every_time("an #include of a file that was not given", "", {"--src", "#include \"local.h\"\n__global__ void k() {}"});
  ::unlink("local.h");
  every_time("an #include through a macro", "", {"--src", "#define F <x.h>\n#include F\n__global__ void k() {}"});
  every_time("an #include on a continued line", "", {"--src", "#inc\\\nlude \"y.h\"\n__global__ void k() {}"});
  every_time("__DATE__", "", {"--src", "const char* d = __DATE__; __global__ void k() {}"});
  every_time("__FILE__", "", {"--src", "const char* f = __FILE__; __global__ void k() {}"});
  every_time("the option -g", "", {"--src", "__global__ void k() {}", "--opt", "-g"});
  every_time("the option -include", "", {"--src", "__global__ void k() {}", "--opt", "-include", "--opt", "x.h"});
  every_time("an option nobody listed", "", {"--src", "__global__ void k() {}", "--opt", "-fplugin=x.so"});
  every_time("a reserved macro", "", {"--src", "__global__ void k() {}", "--opt", "-D__HIP_PLATFORM_NVIDIA__"});
  every_time("an option handed to the preprocessor", "", {"--src", "__global__ void k() {}", "--opt", "-Wp,-MD,deps.d"});
  once("an #include of a header that was given", "", {"--src", "#include \"mine.h\"\n__global__ void k() {}", "--header", "mine.h=#define M 1"});
  once("\"#include\" in a comment", "", {"--src", "// #include \"gone.h\"\n/* #include <gone2.h> */\n__global__ void k() {}"});
  once("an include path with nothing to find through it", "", {"--src", "__global__ void k2() {}", "--opt", "-I/somewhere"});
  // <hip/hip_runtime.h>: nothing, where the built-in header has its guard and the file is the guarded one
  const std::vector<std::string> with_runtime = {"--src", "#include <hip/hip_runtime.h>\n__global__ void k() {}"};
  once("<hip/hip_runtime.h>, the guarded header", "", with_runtime);
  every_time("<hip/hip_runtime.h> with an include path", "", {"--src", "#include <hip/hip_runtime.h>\n__global__ void k() {}", "--opt", "-I/somewhere"});
  every_time("<hip/hip_runtime.h> with CPATH set", "CPATH=/somewhere", with_runtime);
  spit(g_root + "/include/hip/hip_runtime.h", guarded + "#define AFTER_THE_GUARD 1\n");
  every_time("<hip/hip_runtime.h> with something after its guard", "", with_runtime);
  spit(g_root + "/include/hip/hip_runtime.h", "#include <stdint.h>\n" + guarded);
  every_time("<hip/hip_runtime.h> with an #include before its guard", "", with_runtime);
  spit(g_root + "/include/hip/hip_runtime.h", "extern int before_the_guard;\n" + guarded);
  every_time("<hip/hip_runtime.h> with a declaration before its guard", "", with_runtime);
  spit(g_root + "/include/hip/hip_runtime.h", "#ifndef SOME_OTHER_GUARD\n#define SOME_OTHER_GUARD\n#endif\n");
  every_time("<hip/hip_runtime.h> under another guard", "", with_runtime);
  spit(g_root + "/include/hip/hip_runtime.h", guarded + "\n");
  once("<hip/hip_runtime.h> changed and still guarded (a new compile, then kept)", "", with_runtime);
  ::unlink((g_root + "/include/hip/hip_runtime.h").c_str());
  // (a host with ROCm's headers in /usr/include has one the compile could find: the case is this installation's alone)
  if (::access("/usr/include/hip/hip_runtime.h", F_OK) != 0 && ::access("/usr/local/include/hip/hip_runtime.h", F_OK) != 0)
    every_time("<hip/hip_runtime.h> that is nowhere", "", with_runtime);
  spit(g_root + "/include/hip/hip_runtime.h", guarded);
  copy(libs + "/libfake_builtins_b.so", g_root + "/lib/libhiprtc-builtins.so");
  every_time("<hip/hip_runtime.h> under a built-in header without the guard", "", with_runtime);
  copy(libs + "/libfake_builtins_a.so", g_root + "/lib/libhiprtc-builtins.so");
  spit("hiprtc_runtime.h", "// in the way\n");
  every_time("hiprtc_runtime.h in the working directory", "", {"--src", "__global__ void k3() {}"});
  ::unlink("hiprtc_runtime.h");

  // ---- failures, and the calls the cache has no answer for
  reset_cache();
  {
    const std::vector<std::string> bad = {"--src", "__global__ void k() { FAIL }"};
    const std::string w = off("", bad);
    const Outcome a = measure("", bad), b = measure("", bad);
    check(w.find("compile=6") != std::string::npos && a.out == w && b.out == w, "a compile that fails gives the real library's code and log");
    check(a.compiles == 1 && b.compiles == 1 && files_in(g_cache).empty(), "... and is not kept");
  }
  for (const char* how : {"bitcode", "twice", "late-expr", "unknown-name", "null-args", "early"}) {
    const std::vector<std::string> args = {"--src", std::string("__global__ void k() {} // ") + how, "--expr", "k", "--how", how};
    const std::string w = off("", args);
    const Outcome cold = measure("", args), warm = measure("", args);
    check(cold.out == w && warm.out == w, std::string("the calls of \"") + how + "\" give the real library's answers, cold and from the cache");
    if (!std::strcmp(how, "bitcode") || !std::strcmp(how, "twice") || !std::strcmp(how, "late-expr") || !std::strcmp(how, "unknown-name"))
      check(warm.compiles >= 1, "... by compiling for real when the cache has no answer");
  }
  {
    const std::string w = run("VGPU_HIPRTC_CACHE=0", {"--misc"}), c = run("", {"--misc"});
    check(w == c && w.find("version=0 9.7") != std::string::npos && w.find("data m BITCODE;") != std::string::npos,
          "hiprtcVersion and the link API are the real library's");
  }

  // ---- entries that are not whole, or not this user's
  reset_cache();
  const std::string one_want = off("", base);
  measure("", base);
  const std::vector<std::string> entry = files_in(g_cache);
  check(entry.size() == 1, "(one entry to damage)");
  if (entry.size() == 1) {
    const std::string path = g_cache + "/" + entry[0];
    const std::string good = slurp(path);
    const auto damaged = [&](const std::string& what, const std::string& bytes) {
      spit(path, bytes);
      ::chmod(path.c_str(), 0600);
      const Outcome o = measure("", base);
      check(o.compiles == 1 && o.out == one_want, "an entry " + what + " is not used");
      check(slurp(path) == good && measure("", base).compiles == 0, "... and is replaced by a whole one");
    };
    std::string flipped = good;
    flipped[flipped.size() - 40] = static_cast<char>(flipped[flipped.size() - 40] ^ 0x01);
    damaged("with one bit of its code changed", flipped);
    std::string keyflip = good;
    keyflip[100] = static_cast<char>(keyflip[100] ^ 0x20);
    damaged("with one bit of its key changed", keyflip);
    damaged("cut short", good.substr(0, good.size() - 9));
    damaged("cut to its head", good.substr(0, 48));
    damaged("that is empty", "");
    damaged("with bytes after its end", good + "x");
    std::string versioned = good;
    versioned[8] = 2;   // the format's number
    damaged("of another format", versioned);
    std::string other = good;
    other[3] = 'X';
    damaged("that is not an entry", other);
    // Another program's entry under this one's name: its key is not this key.
    // (with as many name expressions as the base program: nothing but its key tells it from the base's)
    const std::vector<std::string> second = {"--src", "__global__ void other() {}", "--expr", "k"};
    measure("", second);
    std::string foreign;
    for (const std::string& f : files_in(g_cache))
      if (f != entry[0]) foreign = slurp(g_cache + "/" + f);
    check(!foreign.empty(), "(a second entry)");
    damaged("that is another program's", foreign);
    // Writable by others: not trusted.
    ::chmod(path.c_str(), 0666);
    check(measure("", base).compiles == 1, "an entry others may write is not used");
  }
  reset_cache();
  ::chmod(g_cache.c_str(), 0777);
  {
    const Outcome a = measure("", base), b = measure("", base);
    check(a.out == one_want && b.out == one_want && a.compiles == 1 && b.compiles == 1 && files_in(g_cache).empty(),
          "a cache directory others may write is not used: compiled every time, nothing written");
  }
  ::chmod(g_cache.c_str(), 0700);

  // ---- eight processes on a cold cache at once
  reset_cache();
  {
    std::vector<std::string> many;
    for (int i = 0; i < 6; ++i) {
      if (i) many.push_back("--next");
      many.push_back("--src"), many.push_back("__global__ void r" + std::to_string(i) + "() {} // SIZE=300000");
    }
    const std::string w = off("", many);
    std::vector<pid_t> kids;
    for (int k = 0; k < 8; ++k) {
      const pid_t pid = ::fork();
      if (pid == 0) {
        const std::string out = run("FAKE_HIPRTC_SLEEP_MS=20", many);
        ::_exit(out == w ? 0 : 1);
      }
      kids.push_back(pid);
    }
    bool all_right = true;
    for (const pid_t pid : kids) {
      int status = 0;
      ::waitpid(pid, &status, 0);
      all_right = all_right && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    check(all_right, "eight processes filling a cold cache at once all get the real library's bytes");
    bool no_tmp = true;
    for (const std::string& f : files_in(g_cache)) no_tmp = no_tmp && f.rfind(".tmp.", 0) != 0;
    check(no_tmp && files_in(g_cache).size() == 6, "... and leave six whole entries and no temporary file");
    const Outcome after = measure("", many);
    check(after.compiles == 0 && after.out == w, "... which the next process uses");
  }

  // ---- the size cap
  reset_cache();
  {
    std::vector<std::string> big;
    for (int i = 0; i < 40; ++i) {
      if (i) big.push_back("--next");
      big.push_back("--src"), big.push_back("__global__ void s" + std::to_string(i) + "() {} // SIZE=200000");
    }
    measure("VGPU_HIPRTC_CACHE_MAX_MB=1", big);
    measure("VGPU_HIPRTC_CACHE_MAX_MB=1", {"--src", "__global__ void last() {} // SIZE=200000"});
    check(bytes_in(g_cache) <= (1u << 20), "the cache stays under VGPU_HIPRTC_CACHE_MAX_MB once a process has written to it");
    check(!files_in(g_cache).empty(), "... and is not emptied");
    reset_cache();
    measure("VGPU_HIPRTC_CACHE_MAX_MB=0", big);
    check(files_in(g_cache).size() == 40, "VGPU_HIPRTC_CACHE_MAX_MB=0 keeps everything");
  }

  // ---- the statistics, the search for the real library, and a real library that is not there
  reset_cache();
  {
    const std::string err = run("VGPU_HIPRTC_CACHE_STATS=1", {"--src", "__global__ void st() {}", "--next", "--src", "__global__ void st() {}"}, true);
    check(err.find("hiprtc-cache: 2 compiles: 1 from memory") != std::string::npos && err.find("1 stored") != std::string::npos,
          "VGPU_HIPRTC_CACHE_STATS=1 says what the cache did");
    const std::string err2 = run("VGPU_HIPRTC_CACHE_STATS=1", {"--src", "__global__ void k() {}", "--opt", "-g"}, true);
    check(err2.find("1 not cacheable: an option outside the list (-g) x1") != std::string::npos, "... and why a program was not cacheable");
    check(run("", {"--src", "__global__ void quiet() {}"}, true).find("hiprtc-cache") == std::string::npos, "... and nothing is printed unasked");
  }
  {
    // Without VGPU_HIPRTC_REAL: the first libhiprtc on the library path that is not the simulator's own, which is in front of it.
    Dl_info info;
    std::string ours = dladdr(reinterpret_cast<void*>(&hiprtcCompileProgram), &info) && info.dli_fname ? info.dli_fname : "";
    ours = ours.substr(0, ours.rfind('/'));
    const std::string cmd = "env -u VGPU_HIPRTC_REAL LD_LIBRARY_PATH=" + sh_quote(ours + ":" + g_root + "/lib") + " VGPU_HIPRTC_CACHE=mem VGPU_HIPRTC_CACHE_STATS=1 " +
                            "VGPU_HIPRTC_CACHE_DIR=" + sh_quote(g_cache) + " " +   // (never the user's own cache, whatever a changed library does)
                            "FAKE_HIPRTC_LOG=" + sh_quote(g_log) + " " + sh_quote(g_self) + " child --src " + sh_quote("__global__ void found() {}") + " 2>&1";
    std::string out;
    if (FILE* f = ::popen(cmd.c_str(), "r")) {
      char buf[1024];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
      ::pclose(f);
    }
    check(out.find("compile=0(FAKE_SUCCESS)") != std::string::npos && out.find("hiprtc-cache: 1 compiles") != std::string::npos,
          "the real library is found on the library path, past the simulator's own");
  }
  {
    const std::string none = run("VGPU_HIPRTC_REAL=/nonexistent/libhiprtc.so.6", {"--src", "__global__ void k() {}"}, true);
    check(none.find("compile=11(HIPRTC_ERROR_INTERNAL_ERROR)") != std::string::npos && none.find("libhiprtc was not found") != std::string::npos,
          "without a real library a compile is an error code and one line that says why");
  }

  // ---- every entry point the real library exports
  {
    Dl_info info;
    bool all_defined = dladdr(reinterpret_cast<void*>(&hiprtcCompileProgram), &info) != 0;
    void* self = all_defined ? ::dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD) : nullptr;
    void* fake = ::dlopen((g_root + "/lib/libhiprtc.so.6").c_str(), RTLD_NOW | RTLD_LOCAL);
    size_t n = 0;
    for (const char* name : {"hiprtcAddNameExpression", "hiprtcCompileProgram", "hiprtcCreateProgram", "hiprtcDestroyProgram", "hiprtcGetBitcode",
                             "hiprtcGetBitcodeSize", "hiprtcGetCode", "hiprtcGetCodeSize", "hiprtcGetErrorString", "hiprtcGetLoweredName",
                             "hiprtcGetProgramLog", "hiprtcGetProgramLogSize", "hiprtcLinkAddData", "hiprtcLinkAddFile", "hiprtcLinkComplete",
                             "hiprtcLinkCreate", "hiprtcLinkDestroy", "hiprtcVersion"}) {
      ++n;
      all_defined = all_defined && self && ::dlsym(self, name) != nullptr && fake && ::dlsym(fake, name) != ::dlsym(self, name);
    }
    check(all_defined && n == 18, "the 18 entry points of ROCm 6.4's and 7's libhiprtc are defined by the simulator's");
  }

  const std::string rm = "rm -rf " + sh_quote(top);
  if (std::system(rm.c_str()) != 0) std::printf("      (could not remove %s)\n", top.c_str());
  std::printf("%s\n", failures ? "FAILED" : "all passed");
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && !std::strcmp(argv[1], "child")) return hiprtc_session::child(argc, argv);
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <directory of the stand-in libraries>\n", argv[0]);
    return 2;
  }
  char self[4096];
  const ssize_t n = ::readlink("/proc/self/exe", self, sizeof self - 1);
  if (n <= 0) {
    std::printf("SKIP: no /proc/self/exe\n");
    return 0;
  }
  g_self = std::string(self, static_cast<size_t>(n));
  // (the working directory is the test's own: two of the cases put a file in it)
  char work[] = "/tmp/vgpu-hiprtc-cwd-XXXXXX";
  if (::mkdtemp(work) && ::chdir(work) == 0) {
    const int r = parent(argv[1]);
    if (::chdir("/") == 0) ::rmdir(work);
    return r;
  }
  return parent(argv[1]);
}
