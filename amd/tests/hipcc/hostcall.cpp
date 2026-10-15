// What a kernel can ask of the host while it runs, through the hostcall buffer
// ROCm's device library speaks (ockl's services.cl; the host's side is
// src/hostcall.cpp): a call to a host function, malloc and free, and the
// printf and fprintf families. Every check prints "<what>: <n> of <m> wrong".
//
//   - __ockl_call_host_function (service 1), declared as ockl.h declares it:
//     a host function's address and seven words go over, two words come
//     back, per lane, from lanes that diverge;
//   - device malloc and free (service 3, __ockl_devmem_request behind them):
//     small blocks used by every lane, a block larger than a slab, blocks
//     freed and allocated again, one too large to exist (a null pointer),
//     and free(NULL);
//   - printf with the conversions C has (the program takes its own stdout
//     and stderr into files while the kernel runs and compares them with the
//     host's printf of the same arguments), and __ockl_fprintf_* to stderr
//     and to stdout, the library's interface that printf is built from.
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// The device library's own entry points (ockl.h), which the compiler links in.
// HIP's device_library_decls.h declares the stderr and append ones (assert is
// built on them); the host function call and stdout's begin are declared here.
typedef long ockl_long2 __attribute__((ext_vector_type(2)));
extern "C" __device__ ockl_long2 __ockl_call_host_function(unsigned long fptr, unsigned long a0, unsigned long a1,
                                                          unsigned long a2, unsigned long a3, unsigned long a4,
                                                          unsigned long a5, unsigned long a6);
extern "C" __device__ unsigned long __ockl_fprintf_stdout_begin();

// ---- The function call service ----
// The host function a kernel names: seven words in, two out.
int g_calls = 0;
extern "C" void host_arithmetic(uint64_t* output, const uint64_t* input) {
  ++g_calls;
  uint64_t sum = 0;
  for (int k = 0; k < 7; ++k) sum += input[k];
  output[0] = sum;
  output[1] = input[0] * 1000 + input[6];
}

__global__ void call_host(unsigned long fn, unsigned long* out) {
  const unsigned long t = threadIdx.x;
  // Lanes that do not call are not in the call: only the others get an answer.
  if (t % 3 != 0) {
    const ockl_long2 r = __ockl_call_host_function(fn, t, 2 * t, 3 * t, 4, 5, 6, 7 + t);
    out[2 * t] = static_cast<unsigned long>(r.x);
    out[2 * t + 1] = static_cast<unsigned long>(r.y);
  }
}

// ---- malloc and free ----
__global__ void small_blocks(unsigned long long* sums, unsigned long long* addresses) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  int* p = static_cast<int*>(malloc(24 * sizeof(int)));
  unsigned long long s = 0;
  if (p) {
    for (int i = 0; i < 24; ++i) p[i] = t * 100 + i;
    for (int i = 0; i < 24; ++i) s += p[i];
  } else {
    s = ~0ull;
  }
  sums[t] = s;
  addresses[t] = reinterpret_cast<unsigned long long>(p);
  free(p);
}

// Blocks freed and allocated again, round after round: each round's block
// is written, read back and freed, so a block handed out twice at once, or a
// free that loses one, shows in the sums or in the count of nulls.
__global__ void churn(unsigned long long* sums, int rounds) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long total = 0;
  for (int r = 0; r < rounds; ++r) {
    const int n = 4 + (r + t) % 40;
    long long* p = static_cast<long long*>(malloc(n * sizeof(long long)));
    if (!p) {
      total += 1ull << 40;
      continue;
    }
    for (int i = 0; i < n; ++i) p[i] = static_cast<long long>(t) * 1000003 + r * 7 + i;
    for (int i = 0; i < n; ++i) total += static_cast<unsigned long long>(p[i] - (static_cast<long long>(t) * 1000003 + r * 7));
    free(p);
  }
  sums[t] = total;
}

// One lane asks for a block bigger than a slab (4 MiB), writes its two ends
// and reads them back; one for more than any device holds.
__global__ void big_blocks(unsigned long long* out) {
  if (threadIdx.x != 0) return;
  char* big = static_cast<char*>(malloc(4u << 20));
  if (big) {
    big[0] = 11;
    big[(4u << 20) - 1] = 22;
    out[0] = static_cast<unsigned long long>(big[0]) * 1000 + big[(4u << 20) - 1];
    out[1] = reinterpret_cast<unsigned long long>(big);
    free(big);
  } else {
    out[0] = ~0ull;
  }
  void* huge = malloc(1ull << 48);
  out[2] = reinterpret_cast<unsigned long long>(huge);
  free(huge);
  free(nullptr);
  out[3] = 1;   // free(NULL) did not hang or fault
}

// ---- printf and fprintf ----
__global__ void say(const char* runtime_text, int* returned) {
  if (threadIdx.x != 0) return;
  returned[0] = printf("[d] %d|%5d|%-5d|%05d|%+d\n", -42, 42, 42, 42, 42);
  returned[1] = printf("[u] %u|%x|%X|%o|%#x\n", 4000000000u, 0xbeefu, 0xbeefu, 8u, 255u);
  returned[2] = printf("[l] %ld|%lu|%lld|%llx\n", -5000000000l, 5000000000ul, -9000000000000ll, 0x123456789abcull);
  returned[3] = printf("[h] %hd|%hhd|%hu\n", static_cast<short>(-3), static_cast<signed char>(-4), static_cast<unsigned short>(65535));
  returned[4] = printf("[f] %f|%.3f|%8.2f|%-8.1f|%e|%g\n", 3.14159, 2.71828, 1.5, 2.25, 12345.678, 0.0001);
  returned[5] = printf("[c] %c%c%c|%5c|%%\n", 'a', 'b', 'c', 'z');
  returned[6] = printf("[s] %s|%10s|%-10s|%.3s\n", "text", "right", "left", "truncated");
  returned[7] = printf("[r] %s|%s\n", runtime_text, runtime_text + 5);
  returned[8] = printf("[w] %*d|%-*d|%.*f\n", 6, 7, 6, 7, 2, 3.14159);
  returned[9] = printf("no arguments at all\n");
}

// Eight lanes, all printing at once, each its own line.
__global__ void crowd() {
  printf("[w] lane %d of %d\n", threadIdx.x, blockDim.x);
}

__global__ void fprintf_ockl(int* returned) {
  if (threadIdx.x != 0) return;
  // stderr: a format, then its arguments (one word each).
  unsigned long d = __ockl_fprintf_stderr_begin();
  static const char fmt[] = "[e] stderr: %d and %s\n";
  d = __ockl_fprintf_append_string_n(d, fmt, sizeof fmt, 0);
  d = __ockl_fprintf_append_args(d, 1, 42, 0, 0, 0, 0, 0, 0, 0);
  static const char word[] = "word";
  returned[0] = static_cast<int>(__ockl_fprintf_append_string_n(d, word, sizeof word, 1));
  // stdout, with the arguments in one call: the format ends the message.
  d = __ockl_fprintf_stdout_begin();
  static const char fmt2[] = "[o] stdout: %d %d %d\n";
  d = __ockl_fprintf_append_string_n(d, fmt2, sizeof fmt2, 0);
  returned[1] = static_cast<int>(__ockl_fprintf_append_args(d, 3, 1, 2, 3, 0, 0, 0, 0, 1));
}

namespace {

void report(const char* what, int wrong, int of) { std::printf("%s: %d of %d wrong\n", what, wrong, of); }

// Takes fd (1 or 2) into a temporary file until read_back() returns the file's text.
struct Capture {
  int fd, saved = -1, file = -1;
  char path[64];
  explicit Capture(int fd_) : fd(fd_) {
    std::snprintf(path, sizeof path, "/tmp/hostcall-%d-%d.txt", static_cast<int>(getpid()), fd_);
    std::fflush(stdout);
    std::fflush(stderr);
    saved = dup(fd);
    file = open(path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    dup2(file, fd);
  }
  std::string read_back() {
    std::fflush(stdout);
    std::fflush(stderr);
    dup2(saved, fd);
    close(saved);
    std::string text;
    lseek(file, 0, SEEK_SET);
    char buf[4096];
    for (ssize_t n; (n = read(file, buf, sizeof buf)) > 0;) text.append(buf, static_cast<size_t>(n));
    close(file);
    unlink(path);
    return text;
  }
};

std::vector<std::string> lines_of(const std::string& text) {
  std::vector<std::string> v;
  size_t at = 0;
  while (at < text.size()) {
    const size_t nl = text.find('\n', at);
    const size_t end = nl == std::string::npos ? text.size() : nl + 1;
    v.push_back(text.substr(at, end - at));
    at = end;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  // With names on the command line, only those parts run (call, malloc, printf, crowd, fprintf).
  const auto wanted = [&](const char* part) {
    if (argc < 2) return true;
    for (int i = 1; i < argc; ++i)
      if (!std::strcmp(argv[i], part)) return true;
    return false;
  };
  // ---- a host function called from a kernel ----
  if (wanted("call")) {
    unsigned long* d = nullptr;
    CHECK(hipMalloc(&d, 64 * 2 * sizeof(unsigned long)));
    CHECK(hipMemset(d, 0xff, 64 * 2 * sizeof(unsigned long)));
    call_host<<<1, 64>>>(reinterpret_cast<unsigned long>(&host_arithmetic), d);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned long> r(128);
    CHECK(hipMemcpy(r.data(), d, r.size() * sizeof(unsigned long), hipMemcpyDeviceToHost));
    int wrong = 0, calls = 0;
    for (unsigned long t = 0; t < 64; ++t) {
      if (t % 3 == 0) {   // a lane that did not call is left as it was
        wrong += r[2 * t] != ~0ul || r[2 * t + 1] != ~0ul;
        continue;
      }
      ++calls;
      wrong += r[2 * t] != 7 * t + 22 || r[2 * t + 1] != t * 1000 + 7 + t;
    }
    wrong += g_calls != calls;   // the host function ran once for each calling lane
    report("a host function called by each lane of a diverged wave", wrong, 64 + 1);
    CHECK(hipFree(d));
  }

  // ---- malloc and free ----
  if (wanted("malloc")) {
    constexpr int N = 256;
    unsigned long long *sums = nullptr, *addrs = nullptr;
    CHECK(hipMalloc(&sums, N * 8));
    CHECK(hipMalloc(&addrs, N * 8));
    small_blocks<<<N / 64, 64>>>(sums, addrs);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned long long> s(N), a(N);
    CHECK(hipMemcpy(s.data(), sums, N * 8, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(a.data(), addrs, N * 8, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int t = 0; t < N; ++t) {
      unsigned long long want = 0;
      for (int i = 0; i < 24; ++i) want += t * 100 + i;
      wrong += s[t] != want || a[t] == 0 || a[t] % 16 != 0;
    }
    // Every lane freed its block in the same instant it was done with it, so
    // blocks may be reused across lanes; the sums catch two lanes sharing
    // one at the same time only if they wrote to it at once, which each lane
    // does within its own loop.
    report("a small block for every lane, written and read back", wrong, N);

    unsigned long long* churned = nullptr;
    CHECK(hipMalloc(&churned, N * 8));
    constexpr int rounds = 40;
    churn<<<N / 64, 64>>>(churned, rounds);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned long long> c(N);
    CHECK(hipMemcpy(c.data(), churned, N * 8, hipMemcpyDeviceToHost));
    wrong = 0;
    for (int t = 0; t < N; ++t) {
      unsigned long long want = 0;
      for (int r = 0; r < rounds; ++r) {
        const int n = 4 + (r + t) % 40;
        for (int i = 0; i < n; ++i) want += static_cast<unsigned long long>(i);
      }
      wrong += c[t] != want;
    }
    report("blocks allocated, used and freed, round after round", wrong, N);

    unsigned long long* big = nullptr;
    CHECK(hipMalloc(&big, 4 * 8));
    CHECK(hipMemset(big, 0, 4 * 8));
    big_blocks<<<1, 64>>>(big);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    unsigned long long b[4];
    CHECK(hipMemcpy(b, big, sizeof b, hipMemcpyDeviceToHost));
    wrong = 0;
    wrong += b[0] != 11 * 1000 + 22;
    wrong += b[1] == 0 || b[1] % 16 != 0;     // a block of its own, as aligned as malloc's are
    wrong += b[2] != 0;                       // too large: a null pointer
    wrong += b[3] != 1;
    report("a block bigger than a slab, one too big for the device, and free(NULL)", wrong, 4);
    CHECK(hipFree(sums));
    CHECK(hipFree(addrs));
    CHECK(hipFree(churned));
    CHECK(hipFree(big));
  }

  // ---- printf ----
  if (wanted("printf")) {
    char* text = nullptr;
    int* returned = nullptr;
    const char runtime_text[] = "text that lives in device memory";
    CHECK(hipMalloc(&text, sizeof runtime_text));
    CHECK(hipMemcpy(text, runtime_text, sizeof runtime_text, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&returned, 16 * sizeof(int)));
    Capture out(1);
    say<<<1, 64>>>(text, returned);
    const hipError_t launched = hipGetLastError();
    const hipError_t synced = hipDeviceSynchronize();
    const std::string got = out.read_back();
    CHECK(launched);
    CHECK(synced);
    char want[16][160];
    int n[16];
    n[0] = std::snprintf(want[0], 160, "[d] %d|%5d|%-5d|%05d|%+d\n", -42, 42, 42, 42, 42);
    n[1] = std::snprintf(want[1], 160, "[u] %u|%x|%X|%o|%#x\n", 4000000000u, 0xbeefu, 0xbeefu, 8u, 255u);
    n[2] = std::snprintf(want[2], 160, "[l] %ld|%lu|%lld|%llx\n", -5000000000l, 5000000000ul, -9000000000000ll,
                         0x123456789abcull);
    n[3] = std::snprintf(want[3], 160, "[h] %hd|%hhd|%hu\n", static_cast<short>(-3), static_cast<signed char>(-4),
                         static_cast<unsigned short>(65535));
    n[4] = std::snprintf(want[4], 160, "[f] %f|%.3f|%8.2f|%-8.1f|%e|%g\n", 3.14159, 2.71828, 1.5, 2.25, 12345.678, 0.0001);
    n[5] = std::snprintf(want[5], 160, "[c] %c%c%c|%5c|%%\n", 'a', 'b', 'c', 'z');
    n[6] = std::snprintf(want[6], 160, "[s] %s|%10s|%-10s|%.3s\n", "text", "right", "left", "truncated");
    n[7] = std::snprintf(want[7], 160, "[r] %s|%s\n", runtime_text, runtime_text + 5);
    n[8] = std::snprintf(want[8], 160, "[w] %*d|%-*d|%.*f\n", 6, 7, 6, 7, 2, 3.14159);
    n[9] = std::snprintf(want[9], 160, "no arguments at all\n");
    std::string expected;
    for (int k = 0; k < 10; ++k) expected += want[k];
    std::vector<int> r(16);
    CHECK(hipMemcpy(r.data(), returned, 16 * sizeof(int), hipMemcpyDeviceToHost));
    int wrong = got != expected;
    for (int k = 0; k < 10; ++k) wrong += r[k] != n[k];
    if (got != expected) std::printf("  got:\n%s  wanted:\n%s", got.c_str(), expected.c_str());
    report("printf's conversions, flags, widths, runtime strings, and what it returns", wrong, 11);
    CHECK(hipFree(text));
    CHECK(hipFree(returned));
  }

  // ---- many lanes printing at once, each its own line ----
  if (wanted("crowd")) {
    Capture out(1);
    crowd<<<2, 64>>>();
    const hipError_t launched = hipGetLastError();
    const hipError_t synced = hipDeviceSynchronize();
    const std::string got = out.read_back();
    CHECK(launched);
    CHECK(synced);
    std::vector<int> seen(2 * 64, 0);
    int wrong = 0;
    for (const std::string& line : lines_of(got)) {
      int lane = -1, of = -1;
      if (std::sscanf(line.c_str(), "[w] lane %d of %d\n", &lane, &of) != 2 || of != 64 || lane < 0 || lane >= 64) {
        ++wrong;
        continue;
      }
      ++seen[lane];
    }
    // Both work-groups print the same lines: each lane's twice.
    for (int lane = 0; lane < 64; ++lane) wrong += seen[lane] != 2;
    report("every lane of two work-groups prints its own whole line", wrong, 64);
  }

  // ---- the fprintf interface the device library has ----
  if (wanted("fprintf")) {
    int* returned = nullptr;
    CHECK(hipMalloc(&returned, 2 * sizeof(int)));
    Capture err(2);
    Capture out(1);
    fprintf_ockl<<<1, 64>>>(returned);
    const hipError_t launched = hipGetLastError();
    const hipError_t synced = hipDeviceSynchronize();
    const std::string got_out = out.read_back();
    const std::string got_err = err.read_back();
    CHECK(launched);
    CHECK(synced);
    int r[2] = {};
    CHECK(hipMemcpy(r, returned, sizeof r, hipMemcpyDeviceToHost));
    const std::string want_err = "[e] stderr: 42 and word\n", want_out = "[o] stdout: 1 2 3\n";
    int wrong = got_err != want_err || got_out != want_out;
    // What the last call of each message returns is the length of its text.
    wrong += r[0] != static_cast<int>(want_err.size()) || r[1] != static_cast<int>(want_out.size());
    if (wrong) std::printf("  stderr %zu bytes, stdout %zu bytes, returned %d and %d\n", got_err.size(), got_out.size(), r[0], r[1]);
    report("fprintf to stderr and to stdout, and what each returns", wrong, 4);
    CHECK(hipFree(returned));
  }
  return 0;
}
