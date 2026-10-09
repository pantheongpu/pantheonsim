// cuFile (GPUDirect Storage) in compatibility mode: the driver's open count
// and staged configuration, file and buffer registration, cuFileRead and
// cuFileWrite between a file and device, pinned, managed and pageable memory,
// the batch API, the stream-ordered API, and the statistics.
//
// The expectations are NVIDIA's: this program also runs against NVIDIA's
// libcufile (CUDA 13.0) on an RTX 3060 without nvidia-fs, and every status
// below is what that library answered. Where the card puts a cuFile status in
// errno instead of the return value (cuFileRead and cuFileWrite return -1),
// so does this check.
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "../../include/vgpu_cufile.h"

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
// A call that returns a CUfileError_t.
#define IS(call, want) check((int)(call).err == (int)(want), #call " -> " #want)
// A data-path call: its return value, and errno when it failed.
#define RET(call, want, want_errno)                                                       \
  do {                                                                                    \
    errno = 0;                                                                            \
    const long long r_ = (long long)(call);                                               \
    const int e_ = errno;                                                                 \
    const bool ok_ = r_ == (long long)(want) && (r_ >= 0 || e_ == (int)(want_errno));     \
    if (!ok_) std::printf("     got %lld errno %d\n", r_, e_);                             \
    check(ok_, #call " -> " #want " (errno " #want_errno ")");                            \
  } while (0)

#define TRACE_USES(where) \
  do { if (std::getenv("CUFILE_TRACE")) std::printf("     uses %ld at %s\n", cuFileUseCount(), where); } while (0)

static std::string scratch(const char* name) {
  const char* dir = std::getenv("TMPDIR");
  return std::string(dir && *dir ? dir : "/tmp") + "/vgpu_cufile_" + std::to_string(getpid()) + "_" + name;
}

static std::vector<unsigned char> pattern(size_t n, unsigned seed) {
  std::vector<unsigned char> v(n);
  uint32_t x = seed * 2654435761u + 12345u;
  for (auto& b : v) {
    x = x * 1664525u + 1013904223u;
    b = (unsigned char)(x >> 24);
  }
  return v;
}

static std::vector<unsigned char> device_bytes(const void* p, size_t n) {
  std::vector<unsigned char> v(n);
  cudaMemcpy(v.data(), p, n, cudaMemcpyDeviceToHost);
  return v;
}

static std::vector<unsigned char> file_bytes(int fd, off_t off, size_t n) {
  std::vector<unsigned char> v(n);
  const ssize_t r = pread(fd, v.data(), n, off);
  v.resize(r < 0 ? 0 : (size_t)r);
  return v;
}

int main() {
  const size_t N = 1 << 20;
  const std::string path = scratch("data"), dirpath = scratch("dir");
  int v = 0;

  // ---- version and the closed driver ----
  IS(cuFileGetVersion(&v), CU_FILE_SUCCESS);
  check(v >= 1000 && v % 10 == 0, "cuFileGetVersion is 1000 * major + 10 * minor");
  IS(cuFileGetVersion(nullptr), CU_FILE_INVALID_VALUE);
  check(cuFileUseCount() == 0, "no driver use before the first call");

  // ---- staged configuration ----
  size_t sz = 0, mn = 0, mx = 0;
  bool b = false;
  char str[64];
  IS(cuFileGetParameterMinMaxValue(CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE, &mn, &mx), CU_FILE_SUCCESS);
  check(mn == 1 && mx == 256, "io_batchsize may be 1 to 256");
  IS(cuFileGetParameterMinMaxValue(CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB, &mn, &mx), CU_FILE_SUCCESS);
  check(mn == 64 && mx == 16384, "max_direct_io_size_kb may be 64 to 16384");
  IS(cuFileGetParameterMinMaxValue(CUFILE_PARAM_PROFILE_STATS, &mn, &mx), CU_FILE_INVALID_VALUE);
  IS(cuFileGetParameterMinMaxValue(CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE, nullptr, &mx), CU_FILE_INVALID_VALUE);
  IS(cuFileGetParameterSizeT((CUFileSizeTConfigParameter_t)12, &sz), CU_FILE_INVALID_VALUE);
  IS(cuFileGetParameterSizeT(CUFILE_PARAM_PROFILE_STATS, nullptr), CU_FILE_INVALID_VALUE);
  IS(cuFileGetParameterString(CUFILE_PARAM_ENV_LOGFILE_PATH, str, sizeof str), CU_FILE_INVALID_VALUE);
  IS(cuFileGetParameterString(CUFILE_PARAM_LOGGING_LEVEL, str, 0), CU_FILE_INVALID_VALUE);
  IS(cuFileSetParameterSizeT(CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE, 64), CU_FILE_SUCCESS);
  IS(cuFileGetParameterSizeT(CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE, &sz), CU_FILE_SUCCESS);
  check(sz == 64, "a staged value reads back before the open");
  IS(cuFileSetParameterBool(CUFILE_PARAM_PROPERTIES_USE_POLL_MODE, true), CU_FILE_SUCCESS);
  IS(cuFileSetParameterString(CUFILE_PARAM_LOGGING_LEVEL, "WARN"), CU_FILE_SUCCESS);
  std::memset(str, 0, sizeof str);
  IS(cuFileGetParameterString(CUFILE_PARAM_LOGGING_LEVEL, str, 4), CU_FILE_INVALID_VALUE);  // no room for the NUL
  IS(cuFileGetParameterString(CUFILE_PARAM_LOGGING_LEVEL, str, 5), CU_FILE_SUCCESS);
  check(std::string(str) == "WARN", "a staged string reads back before the open");

  // ---- opening: explicit, implicit, counted ----
  const int fd = open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
  check(fd >= 0, "scratch file opened");
  CUfileDescr_t d{};
  d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  d.handle.fd = fd;
  CUfileHandle_t fh = nullptr;
  IS(cuFileHandleRegister(&fh, &d), CU_FILE_SUCCESS);
  check(cuFileUseCount() == 1, "registering a file opens the driver once");
  IS(cuFileDriverOpen(), CU_FILE_SUCCESS);
  check(cuFileUseCount() == 2, "an explicit open counts on top of it");
  IS(cuFileGetParameterSizeT(CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE, &sz), CU_FILE_SUCCESS);
  check(sz == 64, "the staged io_batchsize is the running one");
  IS(cuFileGetParameterBool(CUFILE_PARAM_PROPERTIES_USE_POLL_MODE, &b), CU_FILE_SUCCESS);
  check(b, "the staged poll mode is the running one");
  IS(cuFileGetParameterBool(CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE, &b), CU_FILE_SUCCESS);
  check(b, "compatibility mode is allowed");
  IS(cuFileSetParameterSizeT(CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE, 32), CU_FILE_DRIVER_ALREADY_OPEN);
  IS(cuFileSetParameterBool(CUFILE_PARAM_PROPERTIES_USE_POLL_MODE, false), CU_FILE_DRIVER_ALREADY_OPEN);
  CUfileDrvProps_t props{};
  IS(cuFileDriverGetProperties(&props), CU_FILE_SUCCESS);
  IS(cuFileDriverGetProperties(nullptr), CU_FILE_INVALID_VALUE);
  // The properties describe nvidia-fs, which staged parameters do not change.
  check(props.max_batch_io_size == 128, "properties: the configuration's batch size");
  check((props.nvfs.dcontrolflags >> CU_FILE_ALLOW_COMPAT_MODE) & 1, "properties: compatibility mode allowed");
  check(props.nvfs.dstatusflags == 0, "properties: no nvidia-fs file system support");
  check((props.fflags & 0xf) == 0xf, "properties: batch, stream and parallel I/O");
  IS(cuFileDriverSetPollMode(true, 8), CU_FILE_SUCCESS);
  IS(cuFileDriverSetMaxDirectIOSize(1024), CU_FILE_SUCCESS);
  IS(cuFileDriverSetMaxDirectIOSize(0), CU_FILE_DRIVER_UNSUPPORTED_LIMIT);
  IS(cuFileDriverSetMaxDirectIOSize(size_t(1) << 40), CU_FILE_DRIVER_UNSUPPORTED_LIMIT);
  // Nothing is cached or pinned for DMA without nvidia-fs.
  IS(cuFileDriverSetMaxCacheSize(1024), CU_FILE_DRIVER_UNSUPPORTED_LIMIT);
  IS(cuFileDriverSetMaxPinnedMemSize(1024), CU_FILE_DRIVER_UNSUPPORTED_LIMIT);
  size_t bar = 1;
  IS(cuFileGetBARSizeInKB(0, &bar), CU_FILE_SUCCESS);
  check(bar == 0, "no BAR1 mapping in compatibility mode");
  IS(cuFileGetBARSizeInKB(99, &bar), CU_FILE_INVALID_VALUE);

  TRACE_USES("file registration");
  // ---- file registration ----
  CUfileHandle_t fh2 = nullptr;
  IS(cuFileHandleRegister(&fh2, &d), CU_FILE_HANDLE_ALREADY_REGISTERED);
  IS(cuFileHandleRegister(nullptr, &d), CU_FILE_INVALID_VALUE);
  IS(cuFileHandleRegister(&fh2, nullptr), CU_FILE_INVALID_VALUE);
  CUfileDescr_t bad = d;
  bad.handle.fd = -1;
  IS(cuFileHandleRegister(&fh2, &bad), CU_FILE_INVALID_VALUE);
  bad.handle.fd = 987;  // not open
  IS(cuFileHandleRegister(&fh2, &bad), CU_FILE_INVALID_VALUE);
  mkdir(dirpath.c_str(), 0755);
  const int dfd = open(dirpath.c_str(), O_RDONLY);
  bad.handle.fd = dfd;
  IS(cuFileHandleRegister(&fh2, &bad), CU_FILE_INVALID_FILE_TYPE);
  int pp[2] = {-1, -1};
  check(pipe(pp) == 0, "pipe");
  bad.handle.fd = pp[0];
  IS(cuFileHandleRegister(&fh2, &bad), CU_FILE_INVALID_FILE_TYPE);
  bad = d;
  bad.type = CU_FILE_HANDLE_TYPE_OPAQUE_WIN32;
  IS(cuFileHandleRegister(&fh2, &bad), CU_FILE_IO_NOT_SUPPORTED);
  bad.type = (CUfileFileHandleType)7;
  IS(cuFileHandleRegister(&fh2, &bad), CU_FILE_IO_NOT_SUPPORTED);
  const int rofd = open(path.c_str(), O_RDONLY), wofd = open(path.c_str(), O_WRONLY),
            apfd = open(path.c_str(), O_WRONLY | O_APPEND);
  CUfileHandle_t rofh = nullptr, wofh = nullptr, apfh = nullptr;
  bad = d;
  bad.handle.fd = rofd;
  IS(cuFileHandleRegister(&rofh, &bad), CU_FILE_SUCCESS);
  bad.handle.fd = wofd;
  IS(cuFileHandleRegister(&wofh, &bad), CU_FILE_SUCCESS);
  bad.handle.fd = apfd;
  IS(cuFileHandleRegister(&apfh, &bad), CU_FILE_INVALID_FILE_OPEN_FLAG);

  TRACE_USES("buffer registration");
  // ---- buffer registration ----
  unsigned char *dev = nullptr, *dev2 = nullptr, *pinned = nullptr, *managed = nullptr;
  cudaMalloc((void**)&dev, N);
  cudaMalloc((void**)&dev2, N);
  cudaMallocHost((void**)&pinned, N);
  cudaMallocManaged((void**)&managed, N);
  std::vector<unsigned char> pageable(N);
  IS(cuFileBufRegister(dev, N, 0), CU_FILE_SUCCESS);
  IS(cuFileBufRegister(dev, N, 0), CU_FILE_MEMORY_ALREADY_REGISTERED);
  IS(cuFileBufRegister(dev + 4096, 4096, 0), CU_FILE_SUCCESS);  // a range inside another registers
  IS(cuFileBufDeregister(dev + 4096), CU_FILE_SUCCESS);
  IS(cuFileBufRegister(nullptr, N, 0), CU_FILE_INVALID_VALUE);
  IS(cuFileBufRegister(dev2, 0, 0), CU_FILE_INVALID_VALUE);
  IS(cuFileBufRegister(dev2, 2 * N, 0), CU_FILE_CUDA_POINTER_RANGE_ERROR);
  IS(cuFileBufRegister(dev2 + N / 2, N, 0), CU_FILE_CUDA_POINTER_RANGE_ERROR);
  IS(cuFileBufRegister(pinned, N, 0), CU_FILE_SUCCESS);
  IS(cuFileBufRegister(managed, N, 0), CU_FILE_SUCCESS);
  IS(cuFileBufRegister(pageable.data(), N, 0), CU_FILE_SUCCESS);
  IS(cuFileBufDeregister(nullptr), CU_FILE_INVALID_VALUE);
  IS(cuFileBufDeregister(dev2), CU_FILE_MEMORY_NOT_REGISTERED);
  IS(cuFileBufDeregister(pinned), CU_FILE_SUCCESS);
  IS(cuFileBufDeregister(managed), CU_FILE_SUCCESS);
  IS(cuFileBufDeregister(pageable.data()), CU_FILE_SUCCESS);

  TRACE_USES("writes from every kind of memory");
  // ---- writes from every kind of memory, read back with POSIX ----
  const auto p0 = pattern(N, 1), p1 = pattern(N, 2), p2 = pattern(N, 3);
  cudaMemcpy(dev, p0.data(), N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev2, p1.data(), N, cudaMemcpyHostToDevice);
  std::memcpy(pinned, p2.data(), N);
  std::memcpy(managed, p1.data(), N);
  std::memcpy(pageable.data(), p0.data(), N);
  RET(cuFileWrite(fh, dev, 65536, 0, 0), 65536, 0);                 // registered device memory
  RET(cuFileWrite(fh, dev2, 65536, 65536, 4096), 65536, 0);         // unregistered, at an offset
  RET(cuFileWrite(fh, pinned, 4096, 131072, 0), 4096, 0);
  RET(cuFileWrite(fh, managed, 4096, 135168, 100), 4096, 0);
  RET(cuFileWrite(fh, pageable.data(), 4096, 139264, 7), 4096, 0);
  RET(cuFileWrite(fh, dev, 100, 143360, 3), 100, 0);               // unaligned everything
  check(file_bytes(fd, 0, 65536) == std::vector<unsigned char>(p0.begin(), p0.begin() + 65536),
        "registered device memory written to the file");
  check(file_bytes(fd, 65536, 65536) == std::vector<unsigned char>(p1.begin() + 4096, p1.begin() + 69632),
        "unregistered device memory written at a buffer offset");
  check(file_bytes(fd, 131072, 4096) == std::vector<unsigned char>(p2.begin(), p2.begin() + 4096),
        "pinned memory written");
  check(file_bytes(fd, 135168, 4096) == std::vector<unsigned char>(p1.begin() + 100, p1.begin() + 4196),
        "managed memory written");
  check(file_bytes(fd, 139264, 4096) == std::vector<unsigned char>(p0.begin() + 7, p0.begin() + 4103),
        "pageable memory written");
  check(file_bytes(fd, 143360, 100) == std::vector<unsigned char>(p0.begin() + 3, p0.begin() + 103),
        "an unaligned write");
  const off_t file_size = 143460;
  check(lseek(fd, 0, SEEK_END) == file_size, "the file is as long as the writes");

  TRACE_USES("reads into every kind");
  // ---- reads into every kind of memory ----
  cudaMemset(dev2, 0, N);
  RET(cuFileRead(fh, dev2, 65536, 0, 0), 65536, 0);
  check(device_bytes(dev2, 65536) == std::vector<unsigned char>(p0.begin(), p0.begin() + 65536),
        "read into unregistered device memory");
  RET(cuFileRead(fh, dev, 4096, 65536, 8192), 4096, 0);
  check(device_bytes(dev + 8192, 4096) == std::vector<unsigned char>(p1.begin() + 4096, p1.begin() + 8192),
        "read into registered device memory at a buffer offset");
  std::memset(pageable.data(), 0, N);
  RET(cuFileRead(fh, pageable.data(), 4096, 131072, 0), 4096, 0);
  check(std::memcmp(pageable.data(), p2.data(), 4096) == 0, "read into pageable memory");
  RET(cuFileRead(fh, managed, 1000, 1, 5), 1000, 0);
  check(std::memcmp(managed + 5, p0.data() + 1, 1000) == 0, "read into managed memory, unaligned");
  RET(cuFileRead(fh, dev2, 8192, file_size - 100, 0), 100, 0);       // short at the end of the file
  RET(cuFileRead(fh, dev2, 8192, file_size, 0), 0, 0);
  RET(cuFileRead(fh, dev2, 8192, off_t(1) << 30, 0), 0, 0);
  RET(cuFileRead(fh, dev2, 0, 0, 0), 0, 0);
  RET(cuFileWrite(fh, dev2, 0, 0, 0), 0, 0);

  TRACE_USES("the data path");
  // ---- the data path's refusals ----
  RET(cuFileWrite(fh, dev, N, 32768, 4096), -1, CU_FILE_INVALID_MAPPING_RANGE);   // past the registration
  RET(cuFileWrite(fh, dev, 4096, 0, N), -1, CU_FILE_INVALID_MAPPING_RANGE);
  RET(cuFileRead(fh, dev, 8192, 0, N - 4096), -1, CU_FILE_INVALID_MAPPING_RANGE);
  RET(cuFileRead(fh, dev2, 4096, 0, N - 100), -1, EIO);         // unregistered, past the allocation
  RET(cuFileRead(fh, dev2 + N - 100, 4096, 0, 0), -1, EIO);
  RET(cuFileWrite(fh, dev2, 4096, 0, N - 100), -1, EIO);
  RET(cuFileWrite(fh, nullptr, 4096, 0, 0), -1, EINVAL);
  RET(cuFileWrite(fh, dev, 4096, -1, 0), -1, CU_FILE_INVALID_VALUE);
  RET(cuFileWrite(fh, dev, 4096, 0, -1), -1, CU_FILE_INTERNAL_ERROR);
  RET(cuFileWrite(nullptr, dev, 4096, 0, 0), -1, EINVAL);
  RET(cuFileWrite((CUfileHandle_t)0x1234, dev, 4096, 0, 0), -1, EINVAL);  // never a handle
  RET(cuFileWrite(rofh, dev, 4096, 0, 0), -1, EIO);  // the file system refuses it
  RET(cuFileRead(wofh, dev, 4096, 0, 0), -1, CU_FILE_INVALID_FILE_OPEN_FLAG);
  RET(cuFileRead(rofh, dev2, 4096, 0, 0), 4096, 0);
  RET(cuFileWrite(wofh, dev2, 4096, 0, 0), 4096, 0);
  cuFileHandleDeregister(rofh);
  RET(cuFileRead(rofh, dev2, 4096, 0, 0), -1, CU_FILE_HANDLE_NOT_REGISTERED);
  // ... and that failure costs the driver one use (CUDA 13.0's library).
  check(cuFileUseCount() == 1, "a read through a deregistered handle drops a driver use");
  IS(cuFileDriverOpen(), CU_FILE_SUCCESS);
  cuFileHandleDeregister(nullptr);
  cuFileHandleDeregister(wofh);

  TRACE_USES("batches");
  // ---- batches ----
  CUfileBatchHandle_t bh = nullptr;
  IS(cuFileBatchIOSetUp(&bh, 0), CU_FILE_INVALID_VALUE);
  IS(cuFileBatchIOSetUp(&bh, 257), CU_FILE_INVALID_VALUE);
  IS(cuFileBatchIOSetUp(&bh, 4), CU_FILE_SUCCESS);
  CUfileIOParams_t io[4];
  std::memset(io, 0, sizeof io);
  const auto p3 = pattern(4 * 4096, 4);
  cudaMemcpy(dev2, p3.data(), p3.size(), cudaMemcpyHostToDevice);
  for (int i = 0; i < 4; ++i) {
    io[i].mode = CUFILE_BATCH;
    io[i].u.batch.devPtr_base = dev2;
    io[i].u.batch.file_offset = 200000 + i * 4096;
    io[i].u.batch.devPtr_offset = i * 4096;
    io[i].u.batch.size = 4096;
    io[i].fh = fh;
    io[i].opcode = CUFILE_WRITE;
    io[i].cookie = (void*)(intptr_t)(100 + i);
  }
  IS(cuFileBatchIOSubmit(bh, 5, io, 0), CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR);
  IS(cuFileBatchIOSubmit(bh, 0, io, 0), CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR);
  IS(cuFileBatchIOSubmit(bh, 4, io, 0), CU_FILE_SUCCESS);
  CUfileIOEvents_t ev[8];
  unsigned nr = 8;
  struct timespec ts = {5, 0};
  IS(cuFileBatchIOGetStatus(bh, 9, &nr, ev, &ts), CU_FILE_INVALID_VALUE);
  nr = 8;
  IS(cuFileBatchIOGetStatus(bh, 4, &nr, ev, &ts), CU_FILE_SUCCESS);
  std::set<intptr_t> cookies;
  bool all_complete = nr == 4;
  for (unsigned i = 0; i < nr; ++i) {
    cookies.insert((intptr_t)ev[i].cookie);
    all_complete = all_complete && ev[i].status == CUFILE_COMPLETE && ev[i].ret == 4096;
  }
  check(all_complete && cookies == std::set<intptr_t>{100, 101, 102, 103}, "four batched writes complete");
  check(file_bytes(fd, 200000, p3.size()) == p3, "the batched writes are in the file");
  nr = 8;
  IS(cuFileBatchIOGetStatus(bh, 0, &nr, ev, &ts), CU_FILE_SUCCESS);
  check(nr == 0, "nothing more to reap");
  // A batch of reads, one past its allocation and one past the end of file.
  for (int i = 0; i < 4; ++i) {
    io[i].opcode = CUFILE_READ;
    io[i].u.batch.devPtr_offset = N / 2 + i * 4096;
  }
  io[1].u.batch.devPtr_offset = N - 100;
  io[3].u.batch.file_offset = off_t(1) << 30;
  io[2].fh = (CUfileHandle_t)0x1234;
  IS(cuFileBatchIOSubmit(bh, 4, io, 0), CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR);  // an unregistered file
  io[2].fh = fh;
  io[0].mode = (CUfileBatchMode_t)0;
  IS(cuFileBatchIOSubmit(bh, 4, io, 0), CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR);  // not CUFILE_BATCH
  io[0].mode = CUFILE_BATCH;
  IS(cuFileBatchIOSubmit(bh, 4, io, 0), CU_FILE_SUCCESS);
  nr = 4;
  IS(cuFileBatchIOGetStatus(bh, 4, &nr, ev, &ts), CU_FILE_SUCCESS);
  bool shape = nr == 4;
  for (unsigned i = 0; i < nr; ++i) {
    const intptr_t c = (intptr_t)ev[i].cookie;
    const ssize_t r = (ssize_t)ev[i].ret;
    if (c == 101) shape = shape && ev[i].status == CUFILE_FAILED && r == -EIO;
    else if (c == 103) shape = shape && ev[i].status == CUFILE_COMPLETE && r == 0;
    else shape = shape && ev[i].status == CUFILE_COMPLETE && r == 4096;
  }
  check(shape, "a failed copy is CUFILE_FAILED with -EIO, a read past the end completes with 0");
  check(device_bytes(dev2 + N / 2, 4096) == std::vector<unsigned char>(p3.begin(), p3.begin() + 4096),
        "a batched read landed");
  IS(cuFileBatchIOCancel(bh), CU_FILE_SUCCESS);
  cuFileBatchIODestroy(bh);
  IS(cuFileBatchIOSubmit(bh, 1, io, 0), CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR);
  nr = 8;
  IS(cuFileBatchIOGetStatus(bh, 0, &nr, ev, &ts), CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR);
  TRACE_USES("Host memory in a batch.");
  // Host memory in a batch.
  CUfileBatchHandle_t bh2 = nullptr;
  IS(cuFileBatchIOSetUp(&bh2, 1), CU_FILE_SUCCESS);
  std::memset(pageable.data(), 0, 4096);
  io[0].u.batch.devPtr_base = pageable.data();
  io[0].u.batch.devPtr_offset = 0;
  io[0].u.batch.file_offset = 200000;
  IS(cuFileBatchIOSubmit(bh2, 1, io, 0), CU_FILE_SUCCESS);
  nr = 1;
  IS(cuFileBatchIOGetStatus(bh2, 1, &nr, ev, &ts), CU_FILE_SUCCESS);
  check(nr == 1 && ev[0].status == CUFILE_COMPLETE && ev[0].ret == 4096 &&
            std::memcmp(pageable.data(), p3.data(), 4096) == 0,
        "a batched read into pageable memory");
  cuFileBatchIODestroy(bh2);

  TRACE_USES("stream-ordered I/O");
  // ---- stream-ordered I/O ----
  // NVIDIA's library blocks forever in stream memory operations under WSL,
  // so the card run sets VGPU_CUFILE_SKIP_ASYNC; the simulator always runs it.
  if (!std::getenv("VGPU_CUFILE_SKIP_ASYNC")) {
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    IS(cuFileStreamRegister((CUstream)s, 0), CU_FILE_SUCCESS);
    IS(cuFileStreamRegister((CUstream)s, 0), CU_FILE_INVALID_VALUE);  // twice
    IS(cuFileStreamRegister((CUstream)s, 16), CU_FILE_INVALID_VALUE);
    size_t asz = 4096;
    off_t fo = 300000, bo = 4096;
    ssize_t got = -7;
    cudaMemcpyAsync(dev2 + 4096, p2.data(), 4096, cudaMemcpyHostToDevice, s);
    IS(cuFileWriteAsync(fh, dev2, &asz, &fo, &bo, &got, (CUstream)s), CU_FILE_SUCCESS);
    cudaStreamSynchronize(s);
    check(got == 4096, "a stream-ordered write reports its bytes");
    check(file_bytes(fd, 300000, 4096) == std::vector<unsigned char>(p2.begin(), p2.begin() + 4096),
          "the stream-ordered write follows the copy queued before it");
    asz = 4096;
    fo = 0;
    bo = 0;
    got = -7;
    IS(cuFileReadAsync(fh, dev2, &asz, &fo, &bo, &got, (CUstream)s), CU_FILE_SUCCESS);
    std::vector<unsigned char> back(4096);
    cudaMemcpyAsync(back.data(), dev2, 4096, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    check(got == 4096 && back == std::vector<unsigned char>(p0.begin(), p0.begin() + 4096),
          "a stream-ordered read precedes the copy queued after it");
    IS(cuFileReadAsync(fh, dev2, nullptr, &fo, &bo, &got, (CUstream)s), CU_FILE_INVALID_VALUE);
    IS(cuFileReadAsync(fh, dev2, &asz, &fo, &bo, nullptr, (CUstream)s), CU_FILE_INVALID_VALUE);
    IS(cuFileStreamDeregister((CUstream)s), CU_FILE_SUCCESS);
    cudaStreamDestroy(s);
  }

  TRACE_USES("statistics");
  // ---- statistics ----
  IS(cuFileSetStatsLevel(4), CU_FILE_INVALID_VALUE);
  IS(cuFileSetStatsLevel(-1), CU_FILE_INVALID_VALUE);
  IS(cuFileSetStatsLevel(1), CU_FILE_SUCCESS);
  int level = -1;
  IS(cuFileGetStatsLevel(&level), CU_FILE_SUCCESS);
  check(level == 1, "the statistics level reads back");
  CUfileStatsLevel1_t l1{};
  CUfileStatsLevel2_t l2{};
  IS(cuFileStatsStart(), CU_FILE_SUCCESS);
  IS(cuFileStatsReset(), CU_FILE_SUCCESS);
  RET(cuFileRead(fh, dev2, 4096, 0, 0), 4096, 0);
  IS(cuFileGetStatsL1(&l1), CU_FILE_SUCCESS);
  check(l1.read_ops.ok >= 1 && l1.read_bytes >= 4096, "level 1 statistics count a read");
  IS(cuFileGetStatsL2(&l2), CU_FILE_INVALID_VALUE);  // level 2 is not enabled
  IS(cuFileGetStatsL1(nullptr), CU_FILE_INVALID_VALUE);
  IS(cuFileStatsStop(), CU_FILE_SUCCESS);


  // ---- user-space file system handles: registered, never serviced -----------------
  // The operation table is nvidia-fs's RDMA path's; compatibility mode accepts the
  // handle (whatever the table holds, but not none) and then answers every read and
  // write with the number 5006 -- not -1 -- without calling the table or writing the
  // buffer. (This block runs on NVIDIA's library too.)
  {
    static int table_calls = 0;
    struct Ops {
      static ssize_t rd(void*, char* buf, size_t n, loff_t, cufileRDMAInfo_t*) {
        ++table_calls;
        std::memset(buf, 0x5a, n);
        return (ssize_t)n;
      }
      static ssize_t wr(void*, const char*, size_t n, loff_t, cufileRDMAInfo_t*) {
        ++table_calls;
        return (ssize_t)n;
      }
    };
    CUfileFSOps_t table{}, empty{};
    table.read = Ops::rd;
    table.write = Ops::wr;
    const int ufd = open(path.c_str(), O_RDWR);
    CUfileDescr_t ud{};
    ud.type = CU_FILE_HANDLE_TYPE_USERSPACE_FS;
    ud.handle.fd = ufd;
    ud.fs_ops = &table;
    CUfileHandle_t uh = nullptr, uh2 = nullptr;
    CUfileDescr_t none = ud;
    none.fs_ops = nullptr;
    IS(cuFileHandleRegister(&uh2, &none), CU_FILE_IO_NOT_SUPPORTED);
    none.fs_ops = &empty;
    IS(cuFileHandleRegister(&uh2, &none), CU_FILE_SUCCESS);  // a zeroed table registers
    cuFileHandleDeregister(uh2);
    none = ud;
    none.handle.fd = -1;
    IS(cuFileHandleRegister(&uh2, &none), CU_FILE_INVALID_VALUE);
    none.handle.fd = 9999;
    IS(cuFileHandleRegister(&uh2, &none), CU_FILE_INVALID_VALUE);
    IS(cuFileHandleRegister(&uh, &ud), CU_FILE_SUCCESS);
    IS(cuFileHandleRegister(&uh2, &ud), CU_FILE_HANDLE_ALREADY_REGISTERED);
    cudaMemset(dev, 0, 8192);
    errno = 0;
    check(cuFileRead(uh, dev, 4096, 0, 0) == 5006 && errno == 0, "a user-space handle's cuFileRead returns 5006");
    errno = 0;
    check(cuFileWrite(uh, dev, 4096, 0, 0) == 5006 && errno == 0, "a user-space handle's cuFileWrite returns 5006");
    errno = 0;
    check(cuFileRead(uh, dev, 0, 0, 0) == 0, "... a size of 0 on a registered buffer returns 0");
    RET(cuFileRead(uh, nullptr, 4096, 0, 0), -1, EINVAL);
    RET(cuFileRead(uh, dev, 4096, -1, 0), -1, CU_FILE_INVALID_VALUE);
    check(table_calls == 0, "the operation table is never called");
    check(device_bytes(dev, 8)[0] == 0, "the buffer is not written");
    cuFileHandleDeregister(uh);
    close(ufd);
  }

  TRACE_USES("closing");
  // ---- closing ----
  cuFileHandleDeregister(fh);
  IS(cuFileBufDeregister(dev), CU_FILE_SUCCESS);
  long uses = cuFileUseCount();
  std::printf("     use count %ld\n", uses);
  check(uses == 2, "deregistering does not close the driver");
  IS(cuFileDriverClose(), CU_FILE_SUCCESS);
  check(cuFileUseCount() == 1, "a close takes one use away");
  IS(cuFileDriverClose(), CU_FILE_SUCCESS);
  check(cuFileUseCount() == 0, "the last close");
  IS(cuFileDriverClose(), CU_FILE_DRIVER_NOT_INITIALIZED);

  close(fd);
  close(rofd);
  close(wofd);
  close(apfd);
  close(dfd);
  close(pp[0]);
  close(pp[1]);
  unlink(path.c_str());
  rmdir(dirpath.c_str());
  cudaFree(dev);
  cudaFree(dev2);
  cudaFreeHost(pinned);
  cudaFree(managed);
  std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
