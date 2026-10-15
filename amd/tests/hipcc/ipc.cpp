// Memory and events shared between processes, answered as ROCm's HIP answers
// them. Memory: the handles hipIpcGetMemHandle makes for one allocation, each
// asked for separately, all open in another process and reach the same bytes
// (vLLM's custom all-reduce asks for the handle of every buffer its graphs
// capture, the same allocation more than once). Events: an
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
#include <vector>

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
    // Then the memory: two handles for one allocation, 1 MiB of 0x5a.
    struct MemShare { hipIpcMemHandle_t first, second; } shared;
    if (read(to_child[0], &shared, sizeof shared) != sizeof shared) _exit(5);
    char mem[4] = {0, 0, 0, 0};
    void *a = nullptr, *b = nullptr;
    mem[0] = hipIpcOpenMemHandle(&a, shared.first, hipIpcMemLazyEnablePeerAccess) == hipSuccess;
    mem[1] = hipIpcOpenMemHandle(&b, shared.second, hipIpcMemLazyEnablePeerAccess) == hipSuccess;
    if (mem[0]) {
      std::vector<unsigned char> back(1 << 20, 0);
      mem[2] = hipMemcpy(back.data(), a, back.size(), hipMemcpyDeviceToHost) == hipSuccess;
      for (unsigned char c : back) mem[2] = mem[2] && c == 0x5a;
    }
    if (mem[1]) {   // a write through the second handle
      mem[3] = hipMemset(b, 0x77, 4096) == hipSuccess && hipDeviceSynchronize() == hipSuccess;
    }
    if (write(from_child[1], mem, sizeof mem) != sizeof mem) _exit(6);
    if (a) (void)hipIpcCloseMemHandle(a);
    if (b) (void)hipIpcCloseMemHandle(b);
    _exit(0);
  }

  // The parent's ends only: a child that dies then ends the parent's reads, which would wait for ever with
  // the write end open here.
  close(to_child[0]);
  close(from_child[1]);
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
  // Memory: handles for one allocation, asked for twice.
  struct MemShare { hipIpcMemHandle_t first, second; } shared;
  void* buf = nullptr;
  (void)hipMalloc(&buf, 1 << 20);
  (void)hipMemset(buf, 0x5a, 1 << 20);
  const hipError_t first = hipIpcGetMemHandle(&shared.first, buf), second = hipIpcGetMemHandle(&shared.second, buf);
  check(first == hipSuccess && second == hipSuccess, "an allocation's handle can be asked for twice", err(first) + ", " + err(second));
  if (write(to_child[1], &shared, sizeof shared) != sizeof shared) return 2;
  char mem[4] = {0, 0, 0, 0};
  const bool heard_mem = read(from_child[0], mem, sizeof mem) == sizeof mem;
  int status = 1;
  waitpid(pid, &status, 0);
  check(heard && WIFEXITED(status) && WEXITSTATUS(status) == 0, "the other process ran to the end",
        WIFSIGNALED(status) ? "killed by signal " + std::to_string(WTERMSIG(status))
                            : "exit status " + std::to_string(WEXITSTATUS(status)));
  check(result[0], "the other process opens the handle");
  check(result[1], "its wait on the event waits for the record made here");
  check(result[2], "after which the event has happened there");
  check(result[3], "and it destroys its event");
  check(heard_mem && mem[0], "the other process opens the first memory handle");
  check(heard_mem && mem[1], "and the second, made for the same allocation");
  check(heard_mem && mem[2], "the first one reads what was written here");
  check(heard_mem && mem[3], "and a write through the second");
  std::vector<unsigned char> seen(4096, 0);
  (void)hipMemcpy(seen.data(), buf, seen.size(), hipMemcpyDeviceToHost);
  size_t not_written = 0;
  for (unsigned char c : seen) not_written += c != 0x77;
  check(not_written == 0, "reaches this process's memory", std::to_string(not_written) + " of 4096 bytes not written");
  (void)hipFree(buf);
  (void)hipStreamSynchronize(stream);
  check(hipEventQuery(e) == hipSuccess, "the event has happened here too");
  (void)hipEventDestroy(e);
  (void)hipEventDestroy(plain);
  (void)hipStreamDestroy(stream);

  std::printf("ipc: %d checks, %d failed\n", checks, failures);
  return failures ? 1 : 0;
}
