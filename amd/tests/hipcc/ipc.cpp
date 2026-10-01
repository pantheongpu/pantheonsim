// Events shared between processes, answered as ROCm's HIP answers them: an
// interprocess event's handle opens in another process as an event of its
// own, whose waits wait for the records the first process made, and not in
// the process that made it. It forks before HIP is started, as a program
// handing handles to its workers does. Each check prints "ok <what>" or
// "FAIL <what>: <why>", and the last line counts them. Built by build.sh
// with hipcc; run by amd/tests/e2e/run_hipcc.sh.
#include <hip/hip_runtime.h>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

static int checks = 0, failures = 0;
static void check(bool ok, const char* what, const std::string& why = "") {
  ++checks;
  if (ok) {
    std::printf("ok    %s\n", what);
  } else {
    ++failures;
    std::printf("FAIL  %s%s%s\n", what, why.empty() ? "" : ": ", why.c_str());
  }
  std::fflush(stdout);
}
static std::string err(hipError_t e) { return hipGetErrorName(e); }

// Shared with the child: set by the parent's stream just before it records.
static std::atomic<int>* ready = nullptr;
static void slow_then_ready(void*) {
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  ready->store(1);
}

int main() {
  ready = static_cast<std::atomic<int>*>(
      mmap(nullptr, sizeof(std::atomic<int>), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  ready->store(0);
  int to_child[2], from_child[2];
  if (pipe(to_child) != 0 || pipe(from_child) != 0) return 2;
  const pid_t pid = fork();
  if (pid == 0) {
    hipIpcEventHandle_t handle;
    if (read(to_child[0], &handle, sizeof handle) != sizeof handle) _exit(3);
    hipEvent_t e = nullptr;
    char result[4] = {0, 0, 0, 0};
    result[0] = hipIpcOpenEventHandle(&e, handle) == hipSuccess;
    result[1] = e && hipEventSynchronize(e) == hipSuccess && ready->load() == 1;
    result[2] = e && hipEventQuery(e) == hipSuccess;
    result[3] = e && hipEventDestroy(e) == hipSuccess;
    if (write(from_child[1], result, sizeof result) != sizeof result) _exit(4);
    _exit(0);
  }

  hipEvent_t e = nullptr;
  hipError_t got = hipEventCreateWithFlags(&e, hipEventInterprocess);
  check(got == hipErrorInvalidValue, "an interprocess event has to be made without timing", err(got));
  (void)hipEventCreateWithFlags(&e, hipEventInterprocess | hipEventDisableTiming);
  hipIpcEventHandle_t handle;
  got = hipIpcGetEventHandle(&handle, e);
  check(got == hipSuccess, "an interprocess event has a handle", err(got));
  hipEvent_t plain = nullptr;
  (void)hipEventCreate(&plain);
  hipIpcEventHandle_t none;
  got = hipIpcGetEventHandle(&none, plain);
  check(got == hipErrorInvalidConfiguration, "and an event made for this process has none", err(got));
  hipEvent_t mine = nullptr;
  got = hipIpcOpenEventHandle(&mine, handle);
  check(got == hipErrorInvalidContext, "a process does not open its own event's handle", err(got));

  hipStream_t stream;
  (void)hipStreamCreate(&stream);
  (void)hipLaunchHostFunc(stream, slow_then_ready, nullptr);
  (void)hipEventRecord(e, stream);
  if (write(to_child[1], &handle, sizeof handle) != sizeof handle) return 2;
  char result[4] = {0, 0, 0, 0};
  const bool heard = read(from_child[0], result, sizeof result) == sizeof result;
  int status = 1;
  waitpid(pid, &status, 0);
  check(heard && WIFEXITED(status) && WEXITSTATUS(status) == 0, "the other process ran to the end");
  check(result[0], "the other process opens the handle");
  check(result[1], "its wait on the event waits for the record made here");
  check(result[2], "after which the event has happened there");
  check(result[3], "and it destroys its event");
  (void)hipStreamSynchronize(stream);
  check(hipEventQuery(e) == hipSuccess, "the event has happened here too");
  (void)hipEventDestroy(e);
  (void)hipEventDestroy(plain);
  (void)hipStreamDestroy(stream);

  std::printf("ipc: %d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
