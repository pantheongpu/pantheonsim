// libvgpucufile -- VirtualGPU's cuFile (GPUDirect Storage), presented as
// libcufile.so.0.
//
// GPUDirect Storage moves file data between storage and GPU memory with DMA,
// through the nvidia-fs kernel driver. Where that driver is absent, NVIDIA's
// library runs in "compatibility mode": the transfer is an ordinary POSIX
// pread/pwrite staged through host memory. That is the only mode a simulated
// GPU can have, so it is the mode this library implements: a read is a pread
// into a host buffer and a copy into simulated device memory, a write the
// reverse. The data that lands is exactly what NVIDIA's library would land.
//
// Written from NVIDIA's documented cuFile API (the CUDA 13.0 release) with the
// simulator's own declarations (nvidia/include/vgpu_cufile.h). Where the
// documentation leaves a status open, the answer is what NVIDIA's libcufile
// gave on an RTX 3060 in compatibility mode (no nvidia-fs, CUDA 13.0, the
// stock /etc/cufile.json); each such rule says so where it is applied.
//
// Not implemented: I/O through a user-space file system handle. Its operation
// table is called by nvidia-fs's RDMA path, which compatibility mode never
// takes: the card registers such a handle (with a non-null table), then returns
// 5006 from every cuFileRead and cuFileWrite on it without calling the table or
// touching the buffer, and this library does the same. Nor are nvidia-fs's DMA
// paths (GPUDirect proper): there is no kernel module to take them.
#include "../include/vgpu_cufile.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
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

#include "vgpu/memory.hpp"
#include "vgpu/runtime/capture.hpp"
#include "vgpu/runtime/shim_memory.hpp"

#undef cuFileDriverClose

namespace {

// What cuFileGetVersion answers: 1000 * major + 10 * minor. NVIDIA's library
// in CUDA 13.0 (libcufile.so.1.15.1) answers 1150. (CUDA 13.2's, 1.17,
// answers 1170 and differs in places noted below; this library is CUDA 13.0's.)
constexpr int kVersion = 1150;
// The largest batch one handle takes (properties.io_batchsize).
constexpr unsigned kMaxBatch = 128;
// Staging granularity between the file and simulated device memory.
constexpr size_t kStage = size_t(16) << 20;

CUfileError_t status(CUfileOpError e) { return CUfileError_t{e, CUDA_SUCCESS}; }
const CUfileError_t kOk = status(CU_FILE_SUCCESS);

// ---- configuration ---------------------------------------------------------
//
// Parameters are staged while the driver is closed and applied when it opens;
// a set while it is open is CU_FILE_DRIVER_ALREADY_OPEN. Get answers the staged
// value before the open and the running value after it.
constexpr int kSizeParams = 12, kBoolParams = 12, kStringParams = 3;

// The running values: the stock /etc/cufile.json, as NVIDIA's library reported
// them after cuFileDriverOpen on an RTX 3060.
constexpr size_t kRunSize[kSizeParams] = {
    0,      // PROFILE_STATS
    128,    // EXECUTION_MAX_IO_QUEUE_DEPTH
    4,      // EXECUTION_MAX_IO_THREADS
    8192,   // EXECUTION_MIN_IO_THRESHOLD_SIZE_KB
    4,      // EXECUTION_MAX_REQUEST_PARALLELISM
    1024,   // PROPERTIES_MAX_DIRECT_IO_SIZE_KB
    131072, // PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB
    1024,   // PROPERTIES_PER_BUFFER_CACHE_SIZE_KB
    33554432, // PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB
    128,    // PROPERTIES_IO_BATCHSIZE
    4,      // POLLTHRESHOLD_SIZE_KB
    5,      // PROPERTIES_BATCH_IO_TIMEOUT_MS
};
// Before the first open, with nothing staged, what is answered: the
// configuration's values (nothing has been read yet that could differ).
constexpr const size_t* kIdleSize = kRunSize;
// The settable range of each, from cuFileGetParameterMinMaxValue on the card
// (CUDA 13.0's library, including its [0, 0] for the device cache and
// [64, 0] for pinned memory). PROFILE_STATS has none: the call is
// CU_FILE_INVALID_VALUE.
constexpr size_t kMin[kSizeParams] = {0, 1, 0, 4, 0, 64, 0, 64, 64, 1, 4, 1};
constexpr size_t kMax[kSizeParams] = {0, 256, 32, 4294967295ull, 8, 16384, 0, 16384,
                                      0, 256, 18014398509481980ull, 1000};
// Whether a staged value is in range when the driver opens. The two ranges
// the library reports with a zero maximum are taken as unbounded.
bool in_range(int p, size_t v) { return v >= kMin[p] && (kMax[p] == 0 || v <= kMax[p]); }

// Compatibility mode allowed, parallel I/O, system memory allowed.
constexpr bool kRunBool[kBoolParams] = {false, true, false, false, true, false,
                                        true, false, false, false, false, false};
constexpr const bool* kIdleBool = kRunBool;
// CUFILE_PARAM_ENV_LOGFILE_PATH is not readable on the card
// (CU_FILE_INVALID_VALUE). (CUDA 13.2's library also refuses PROFILE_NVTX.)
bool bool_readable(int) { return true; }
bool string_readable(int p) { return p != CUFILE_PARAM_ENV_LOGFILE_PATH; }

struct Handle {
  int fd = -1;
  int accmode = O_RDWR;
  bool userspace = false;   // a CU_FILE_HANDLE_TYPE_USERSPACE_FS handle: registers, never does I/O
};

struct BatchEntry {
  void* cookie = nullptr;
  CUfileStatus_t status = CUFILE_WAITING;
  ssize_t ret = 0;
};

struct Batch {
  unsigned capacity = 0;
  unsigned outstanding = 0;                 // submitted and not yet reaped
  std::deque<BatchEntry> done;              // completed, waiting for GetStatus
};

struct Stats {
  CUfileStatsLevel1_t l1{};
  uint64_t read_hist[32] = {};
  uint64_t write_hist[32] = {};
};

struct State {
  std::recursive_mutex mu;
  long use_count = 0;
  bool was_open = false;  // opened once and closed since: CU_FILE_DRIVER_CLOSING
  // Staged configuration (set while closed) and the running configuration.
  bool staged_size_set[kSizeParams] = {};
  size_t staged_size[kSizeParams] = {};
  bool staged_bool_set[kBoolParams] = {};
  bool staged_bool[kBoolParams] = {};
  bool staged_string_set[kStringParams] = {};
  std::string staged_string[kStringParams];
  size_t run_size[kSizeParams] = {};
  bool run_bool[kBoolParams] = {};
  std::string run_string[kStringParams];
  CUfileDrvProps_t props{};
  int stats_level = 0;
  bool stats_on = true;
  Stats stats;
  std::set<Handle*> handles;
  std::set<CUfileHandle_t> retired;  // handles given out and deregistered since
  std::map<uintptr_t, size_t> bufs;  // registered base -> length
  std::set<Batch*> batches;
  std::map<CUstream, unsigned> streams;
};

State& st() {
  static State* s = new State;  // outlives every static destructor that might call in
  return *s;
}

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

void apply_open(State& s) {
  for (int p = 0; p < kSizeParams; ++p) {
    size_t v = kRunSize[p];
    if (s.staged_size_set[p]) {
      const size_t want = s.staged_size[p];
      // A staged value outside the range is reported and replaced by the
      // configuration's own, as NVIDIA's library does when it opens.
      if (p == CUFILE_PARAM_PROFILE_STATS || in_range(p, want)) {
        v = want;
      } else if (!quiet()) {
        std::fprintf(stderr, "[vgpu] cuFileDriverOpen: staged parameter %d = %zu is outside [%zu, %zu]; "
                             "using %zu\n", p, want, kMin[p], kMax[p], v);
      }
    }
    s.run_size[p] = v;
    s.staged_size_set[p] = false;
  }
  for (int p = 0; p < kBoolParams; ++p) {
    s.run_bool[p] = s.staged_bool_set[p] ? s.staged_bool[p] : kRunBool[p];
    s.staged_bool_set[p] = false;
  }
  static const char* kRunString[kStringParams] = {"ERROR", "", ""};
  for (int p = 0; p < kStringParams; ++p) {
    s.run_string[p] = s.staged_string_set[p] ? s.staged_string[p] : kRunString[p];
    s.staged_string_set[p] = false;
  }
  // Driver properties, as NVIDIA's library reports them in compatibility mode
  // on an RTX 3060: no nvidia-fs (version 0.0, no file system flags), compat
  // mode allowed (control bit 1), all four feature flags, and the stock
  // configuration's sizes. They describe the nvidia-fs driver, so staged
  // parameters do not show here (a staged io_batchsize of 64 still reads 128,
  // a staged poll mode leaves bit 0 clear), and the per-buffer cache is 0.
  CUfileDrvProps_t& p = s.props;
  p = CUfileDrvProps_t{};
  p.nvfs.poll_thresh_size = kRunSize[CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB];
  p.nvfs.max_direct_io_size = kRunSize[CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB];
  p.nvfs.dstatusflags = 0;
  p.nvfs.dcontrolflags = 1u << CU_FILE_ALLOW_COMPAT_MODE;
  p.fflags = 0xf;
  p.max_device_cache_size = (unsigned)kRunSize[CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB];
  p.per_buffer_cache_size = 0;
  p.max_device_pinned_mem_size = (unsigned)kRunSize[CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB];
  p.max_batch_io_size = (unsigned)kRunSize[CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE];
  p.max_batch_io_timeout_msecs = (unsigned)kRunSize[CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS];
  s.stats_level = (int)std::min<size_t>(s.run_size[CUFILE_PARAM_PROFILE_STATS], 3);
}

void open_locked(State& s) {
  if (s.use_count == 0) apply_open(s);
  ++s.use_count;
  s.was_open = false;
}

// The calls that need a driver open it when nobody has: on the card,
// cuFileHandleRegister or cuFileBufRegister before cuFileDriverOpen leave
// cuFileUseCount at 1, and a later explicit open makes it 2.
void ensure_open(State& s) {
  if (s.use_count == 0) open_locked(s);
}

void release_all(State& s) {
  for (Handle* h : s.handles) {
    s.retired.insert(h);
    delete h;
  }
  s.handles.clear();
  s.bufs.clear();
  for (Batch* b : s.batches) delete b;
  s.batches.clear();
  s.streams.clear();
}

Handle* find_handle(State& s, CUfileHandle_t fh) {
  auto it = s.handles.find(static_cast<Handle*>(fh));
  return it == s.handles.end() ? nullptr : *it;
}

// ---- moving bytes ----------------------------------------------------------

// Simulated device memory goes through the runtime, which knows where each
// allocation lives; everything else is host memory, which the file system
// reads and writes in place, as compatibility mode does for host buffers.
bool is_device(const void* p) { return vgpu::is_device_va(reinterpret_cast<uint64_t>(p)); }

// Whether [p, p + n) lies inside one device allocation.
bool device_range_ok(const void* p, size_t n) {
  void* base = nullptr;
  size_t size = 0;
  if (!vgpu_device_allocation(p, &base, &size)) return false;
  const uintptr_t off = reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(base);
  return off <= size && n <= size - off;
}

int errno_to_ret(int e) {
  // A bad host address is EFAULT from the kernel; NVIDIA's library reports a
  // buffer it cannot copy as EIO (measured with an unmapped address).
  return e == EFAULT ? EIO : e;
}

// The transfer itself. Returns the byte count, or -1 with errno set.
ssize_t transfer(const Handle& h, char* buf, size_t size, off_t file_offset, bool write) {
  if (is_device(buf)) {
    // NVIDIA's library copies through a bounce buffer and fails the whole
    // call with EIO when the device side is out of range (an RTX 3060:
    // offset + size past the allocation, or an address no allocation holds).
    if (!device_range_ok(buf, size)) {
      errno = EIO;
      return -1;
    }
    std::vector<char> stage(std::min(size, kStage));
    size_t done = 0;
    while (done < size) {
      const size_t want = std::min(size - done, stage.size());
      if (write) {
        if (cudaMemcpy(stage.data(), buf + done, want, cudaMemcpyDefault) != cudaSuccess) {
          cudaGetLastError();
          errno = EIO;
          return -1;
        }
        size_t put = 0;
        while (put < want) {
          const ssize_t w = ::pwrite(h.fd, stage.data() + put, want - put, file_offset + (off_t)(done + put));
          if (w < 0) {
            if (errno == EINTR) continue;
            errno = errno_to_ret(errno);
            return -1;
          }
          put += (size_t)w;
        }
        done += want;
      } else {
        size_t got = 0;
        while (got < want) {
          const ssize_t r = ::pread(h.fd, stage.data() + got, want - got, file_offset + (off_t)(done + got));
          if (r < 0) {
            if (errno == EINTR) continue;
            errno = errno_to_ret(errno);
            return -1;
          }
          if (r == 0) break;  // end of file
          got += (size_t)r;
        }
        if (got && cudaMemcpy(buf + done, stage.data(), got, cudaMemcpyDefault) != cudaSuccess) {
          cudaGetLastError();
          errno = EIO;
          return -1;
        }
        done += got;
        if (got < want) break;
      }
    }
    return (ssize_t)done;
  }
  size_t done = 0;
  while (done < size) {
    const ssize_t r = write ? ::pwrite(h.fd, buf + done, size - done, file_offset + (off_t)done)
                            : ::pread(h.fd, buf + done, size - done, file_offset + (off_t)done);
    if (r < 0) {
      if (errno == EINTR) continue;
      errno = errno_to_ret(errno);
      return -1;
    }
    if (r == 0) break;
    done += (size_t)r;
  }
  return (ssize_t)done;
}

void count(State& s, bool write, ssize_t r, size_t size) {
  if (!s.stats_on || s.stats_level < 1) return;
  CUfileOpCounter_t& c = write ? s.stats.l1.write_ops : s.stats.l1.read_ops;
  if (r < 0) {
    ++c.err;
    return;
  }
  ++c.ok;
  (write ? s.stats.l1.write_bytes : s.stats.l1.read_bytes) += (uint64_t)r;
  int bin = 0;
  for (size_t kb = size >> 10; kb > 1 && bin < 31; kb >>= 1) ++bin;
  (write ? s.stats.write_hist : s.stats.read_hist)[bin]++;
}

// A synchronous read or write, with the checks NVIDIA's library makes, in its
// order. A failure is -1 with errno: a POSIX error, or a cuFile status (the
// card puts CU_FILE_* codes in errno for these, not in the return value).
ssize_t io(CUfileHandle_t fh, void* bufPtr_base, size_t size, off_t file_offset, off_t buf_offset,
           bool write) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  auto fail = [&](int e) -> ssize_t {
    count(s, write, -1, size);
    errno = e;
    return -1;
  };
  // A handle that was deregistered is CU_FILE_HANDLE_NOT_REGISTERED; one
  // this library never gave out is EINVAL, like a null one (CUDA 13.0's
  // library; 13.2's answers CU_FILE_HANDLE_NOT_REGISTERED for both).
  if (!fh) return fail(EINVAL);
  Handle* h = find_handle(s, fh);
  if (!h) {
    if (!s.retired.count(fh)) return fail(EINVAL);
    // CUDA 13.0's library also gives up one use of the driver here: after a
    // read through a deregistered handle cuFileUseCount is one lower (an RTX
    // 3060, 2 -> 1). Matched, so that a program's open/close accounting
    // comes out the same on both.
    if (s.use_count > 0 && --s.use_count == 0) {
      release_all(s);
      s.was_open = true;
    }
    return fail(CU_FILE_HANDLE_NOT_REGISTERED);
  }
  if (!bufPtr_base) return fail(EINVAL);
  // A user-space file system handle registers, but without nvidia-fs its I/O does
  // not run: the card returns 5006 (the number of CU_FILE_DRIVER_CLOSING, as the
  // value of the call, not -1 with errno), without calling the table, touching the
  // buffer or setting errno. A negative offset is checked first; a size of 0 on a
  // registered buffer is 0 as for any handle, on an unregistered one 5006.
  if (h->userspace) {
    if (file_offset < 0) return fail(CU_FILE_INVALID_VALUE);
    if (size == 0 && s.bufs.count(reinterpret_cast<uintptr_t>(bufPtr_base))) return 0;
    static bool said = false;
    if (!said && !quiet()) {
      said = true;
      std::fprintf(stderr, "[vgpu] cuFileRead/cuFileWrite: user-space file system handles do no I/O in "
                           "compatibility mode (they need nvidia-fs); the call returns 5006 as the card's does\n");
    }
    count(s, write, -1, size);
    return static_cast<ssize_t>(5006);
  }
  if (size == 0) return 0;
  if (file_offset < 0) return fail(CU_FILE_INVALID_VALUE);
  if (buf_offset < 0) return fail(CU_FILE_INTERNAL_ERROR);
  // A write-only handle cannot be read: CU_FILE_INVALID_FILE_OPEN_FLAG. A
  // read-only one is not checked; its write fails in the file system, and
  // that is reported as EIO (CUDA 13.0's library; 13.2's checks both).
  if (!write && h->accmode == O_WRONLY) return fail(CU_FILE_INVALID_FILE_OPEN_FLAG);
  if (write && h->accmode == O_RDONLY) return fail(EIO);
  // A registered buffer is looked up by its base: an I/O that runs past the
  // registered length is CU_FILE_INVALID_MAPPING_RANGE (an RTX 3060).
  auto reg = s.bufs.find(reinterpret_cast<uintptr_t>(bufPtr_base));
  if (reg != s.bufs.end() && ((size_t)buf_offset > reg->second || size > reg->second - (size_t)buf_offset))
    return fail(CU_FILE_INVALID_MAPPING_RANGE);
  const ssize_t r = transfer(*h, static_cast<char*>(bufPtr_base) + buf_offset, size, file_offset, write);
  const int e = errno;
  count(s, write, r, size);
  errno = e;
  return r;
}

}  // namespace

#define CUFILE_EXPORT extern "C" __attribute__((visibility("default")))

// ---- driver ----------------------------------------------------------------

CUFILE_EXPORT CUfileError_t cuFileDriverOpen(void) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  open_locked(s);
  return kOk;
}

static CUfileError_t driver_close() {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count == 0) return status(CU_FILE_DRIVER_NOT_INITIALIZED);
  if (--s.use_count == 0) {
    release_all(s);
    s.was_open = true;
  }
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileDriverClose(void) { return driver_close(); }
CUFILE_EXPORT CUfileError_t cuFileDriverClose_v2(void) { return driver_close(); }

CUFILE_EXPORT long cuFileUseCount(void) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  return s.use_count;
}

CUFILE_EXPORT CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t* props) {
  if (!props) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  ensure_open(s);
  *props = s.props;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileDriverSetPollMode(bool poll, size_t poll_threshold_size) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count == 0) return status(CU_FILE_DRIVER_NOT_INITIALIZED);
  (void)poll;
  (void)poll_threshold_size;  // compatibility mode never polls nvidia-fs
  return kOk;
}

// The size setters take KiB and describe nvidia-fs's DMA. The direct I/O
// size takes a value in its parameter's range; the cache and pinned-memory
// limits have nothing to limit in compatibility mode, and the card refuses
// every value for them. All refusals are CU_FILE_DRIVER_UNSUPPORTED_LIMIT
// (CUDA 13.0's library; 13.2's says CU_FILE_INVALID_VALUE for a direct I/O
// size of 0 and accepts the other two).
static CUfileError_t set_limit(int param, size_t kb) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count == 0) return status(CU_FILE_DRIVER_NOT_INITIALIZED);
  if (param != CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB || !in_range(param, kb))
    return status(CU_FILE_DRIVER_UNSUPPORTED_LIMIT);
  return kOk;
}
CUFILE_EXPORT CUfileError_t cuFileDriverSetMaxDirectIOSize(size_t kb) {
  return set_limit(CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB, kb);
}
CUFILE_EXPORT CUfileError_t cuFileDriverSetMaxCacheSize(size_t kb) {
  return set_limit(CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB, kb);
}
CUFILE_EXPORT CUfileError_t cuFileDriverSetMaxPinnedMemSize(size_t kb) {
  return set_limit(CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB, kb);
}

CUFILE_EXPORT CUfileError_t cuFileGetVersion(int* version) {
  if (!version) return status(CU_FILE_INVALID_VALUE);
  *version = kVersion;
  return kOk;
}

// The BAR1 aperture nvidia-fs maps buffers through, in KiB. Compatibility
// mode maps nothing, and CUDA 13.0's library answers 0 for each GPU there; a
// GPU that does not exist is CU_FILE_INVALID_VALUE.
CUFILE_EXPORT CUfileError_t cuFileGetBARSizeInKB(int gpuIndex, size_t* barSize) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();
    n = 0;
  }
  if (!barSize || gpuIndex < 0 || gpuIndex >= n) return status(CU_FILE_INVALID_VALUE);
  *barSize = 0;
  return kOk;
}

// ---- files -------------------------------------------------------------------

CUFILE_EXPORT CUfileError_t cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr) {
  if (!fh || !descr) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  ensure_open(s);
  // A Linux file descriptor, or a user-space file system's (a table of read and
  // write callbacks over a descriptor, for nvidia-fs's RDMA path). The card
  // registers the second whatever the table holds -- even zeroed -- but a null
  // table is CU_FILE_IO_NOT_SUPPORTED, and so are a Windows handle and an unknown
  // type. Its I/O does not work without nvidia-fs (see io below).
  bool userspace = false;
  if (descr->type == CU_FILE_HANDLE_TYPE_USERSPACE_FS) {
    if (!descr->fs_ops) return status(CU_FILE_IO_NOT_SUPPORTED);
    userspace = true;
  } else if (descr->type != CU_FILE_HANDLE_TYPE_OPAQUE_FD) {
    return status(CU_FILE_IO_NOT_SUPPORTED);
  }
  const int fd = descr->handle.fd;
  struct stat sb {};
  if (fd < 0 || ::fstat(fd, &sb) != 0) return status(CU_FILE_INVALID_VALUE);
  // A directory or a pipe is CU_FILE_INVALID_FILE_TYPE; a block device is
  // storage cuFile reads like a file.
  if (!S_ISREG(sb.st_mode) && !S_ISBLK(sb.st_mode)) return status(CU_FILE_INVALID_FILE_TYPE);
  const int fl = ::fcntl(fd, F_GETFL);
  if (fl < 0) return status(CU_FILE_INVALID_VALUE);
  // An append-only descriptor ignores the offsets cuFile writes at: refused
  // as CU_FILE_INVALID_FILE_OPEN_FLAG on the card.
  if (fl & O_APPEND) return status(CU_FILE_INVALID_FILE_OPEN_FLAG);
  for (Handle* h : s.handles)
    if (h->fd == fd) return status(CU_FILE_HANDLE_ALREADY_REGISTERED);
  Handle* h = new Handle;
  h->fd = fd;
  h->accmode = fl & O_ACCMODE;
  h->userspace = userspace;
  s.handles.insert(h);
  if (s.stats_level >= 1) ++s.stats.l1.hdl_register_ops.ok;
  *fh = h;
  return kOk;
}

CUFILE_EXPORT void cuFileHandleDeregister(CUfileHandle_t fh) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  Handle* h = find_handle(s, fh);
  if (!h) return;
  s.handles.erase(h);
  s.retired.insert(fh);
  delete h;
  if (s.stats_level >= 1) ++s.stats.l1.hdl_deregister_ops.ok;
}

// ---- buffers ------------------------------------------------------------------

// Registration pins a buffer for DMA. Here it records the range, which decides
// the CU_FILE_INVALID_MAPPING_RANGE answer of a later I/O. The rules are the
// card's: a base registered twice is CU_FILE_MEMORY_ALREADY_REGISTERED, but a
// range inside another registration is a registration of its own; device
// memory must lie in its allocation (CU_FILE_CUDA_POINTER_RANGE_ERROR); host
// memory of any kind is accepted, unchecked; the flags are not checked.
CUFILE_EXPORT CUfileError_t cuFileBufRegister(const void* bufPtr_base, size_t length, int flags) {
  (void)flags;
  if (!bufPtr_base || length == 0) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  ensure_open(s);
  const uintptr_t key = reinterpret_cast<uintptr_t>(bufPtr_base);
  if (s.bufs.count(key)) return status(CU_FILE_MEMORY_ALREADY_REGISTERED);
  if (is_device(bufPtr_base) && !device_range_ok(bufPtr_base, length)) {
    if (s.stats_level >= 1) ++s.stats.l1.buf_register_ops.err;
    return status(CU_FILE_CUDA_POINTER_RANGE_ERROR);
  }
  s.bufs[key] = length;
  if (s.stats_level >= 1) ++s.stats.l1.buf_register_ops.ok;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileBufDeregister(const void* bufPtr_base) {
  if (!bufPtr_base) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  // After the last close everything was released; the card answers
  // CU_FILE_DRIVER_CLOSING.
  if (s.use_count == 0) return status(s.was_open ? CU_FILE_DRIVER_CLOSING : CU_FILE_DRIVER_NOT_INITIALIZED);
  auto it = s.bufs.find(reinterpret_cast<uintptr_t>(bufPtr_base));
  if (it == s.bufs.end()) return status(CU_FILE_MEMORY_NOT_REGISTERED);
  s.bufs.erase(it);
  if (s.stats_level >= 1) ++s.stats.l1.buf_deregister_ops.ok;
  return kOk;
}

// ---- synchronous I/O -----------------------------------------------------------

CUFILE_EXPORT ssize_t cuFileRead(CUfileHandle_t fh, void* bufPtr_base, size_t size, off_t file_offset,
                                 off_t bufPtr_offset) {
  return io(fh, bufPtr_base, size, file_offset, bufPtr_offset, false);
}

CUFILE_EXPORT ssize_t cuFileWrite(CUfileHandle_t fh, const void* bufPtr_base, size_t size, off_t file_offset,
                                  off_t bufPtr_offset) {
  return io(fh, const_cast<void*>(bufPtr_base), size, file_offset, bufPtr_offset, true);
}

// ---- batch I/O -----------------------------------------------------------------
//
// A submitted batch runs at once, in order; its completions wait in the batch
// until cuFileBatchIOGetStatus reaps them. NVIDIA's library completes them on
// worker threads, in any order, so a program cannot tell the difference.

CUFILE_EXPORT CUfileError_t cuFileBatchIOSetUp(CUfileBatchHandle_t* batch_idp, unsigned nr) {
  // A null handle pointer crashes NVIDIA's library; refuse it here.
  if (!batch_idp) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  ensure_open(s);
  if (nr == 0 || nr > s.props.max_batch_io_size || nr > kMaxBatch) return status(CU_FILE_INVALID_VALUE);
  Batch* b = new Batch;
  b->capacity = nr;
  s.batches.insert(b);
  if (s.stats_level >= 1) ++s.stats.l1.batch_setup_ops.ok;
  *batch_idp = b;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileBatchIOSubmit(CUfileBatchHandle_t batch_idp, unsigned nr, CUfileIOParams_t* iocbp,
                                                unsigned int flags) {
  (void)flags;
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  auto it = s.batches.find(static_cast<Batch*>(batch_idp));
  // The card answers every refused submission -- an unknown or destroyed
  // batch, no entries, more than the batch has room for, an entry that is not
  // CUFILE_BATCH mode or names an unregistered file -- with
  // CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR, and runs none of it.
  const CUfileError_t refused = status(CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR);
  if (it == s.batches.end() || !iocbp || nr == 0) return refused;
  Batch* b = *it;
  if (nr > b->capacity - b->outstanding) return refused;
  for (unsigned i = 0; i < nr; ++i) {
    const CUfileIOParams_t& p = iocbp[i];
    if (p.mode != CUFILE_BATCH || !find_handle(s, p.fh)) return refused;
  }
  for (unsigned i = 0; i < nr; ++i) {
    const CUfileIOParams_t& p = iocbp[i];
    BatchEntry e;
    e.cookie = p.cookie;
    // An opcode that is neither read nor write is accepted and fails.
    if (p.opcode != CUFILE_READ && p.opcode != CUFILE_WRITE) {
      e.status = CUFILE_FAILED;
      e.ret = -EINVAL;
    } else {
      const ssize_t r = io(p.fh, p.u.batch.devPtr_base, p.u.batch.size, p.u.batch.file_offset,
                           p.u.batch.devPtr_offset, p.opcode == CUFILE_WRITE);
      if (r < 0) {
        e.status = CUFILE_FAILED;
        e.ret = -(ssize_t)errno;  // the card: ret is -EIO for a failed copy
      } else {
        e.status = CUFILE_COMPLETE;
        e.ret = r;
      }
    }
    b->done.push_back(e);
    ++b->outstanding;
  }
  if (s.stats_level >= 1) ++s.stats.l1.batch_submit_ops.ok;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileBatchIOGetStatus(CUfileBatchHandle_t batch_idp, unsigned min_nr, unsigned* nr,
                                                   CUfileIOEvents_t* iocbp, struct timespec* timeout) {
  (void)timeout;  // everything submitted has completed already
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  auto it = s.batches.find(static_cast<Batch*>(batch_idp));
  if (it == s.batches.end()) return status(CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR);
  if (!nr || (*nr && !iocbp)) return status(CU_FILE_INVALID_VALUE);
  // Asking for more completions than there is room for is CU_FILE_INVALID_VALUE.
  if (min_nr > *nr) return status(CU_FILE_INVALID_VALUE);
  Batch* b = *it;
  unsigned n = 0;
  while (n < *nr && !b->done.empty()) {
    const BatchEntry& e = b->done.front();
    iocbp[n].cookie = e.cookie;
    iocbp[n].status = e.status;
    iocbp[n].ret = (size_t)e.ret;
    b->done.pop_front();
    --b->outstanding;
    ++n;
  }
  *nr = n;
  if (s.stats_level >= 1) s.stats.l1.batch_complete_ops.ok += n;
  return kOk;
}

// Nothing is ever in flight, so there is nothing to cancel: the completions
// stay to be reaped. A null or unknown batch is a success on the card.
CUFILE_EXPORT CUfileError_t cuFileBatchIOCancel(CUfileBatchHandle_t batch_idp) {
  (void)batch_idp;
  return kOk;
}

CUFILE_EXPORT void cuFileBatchIODestroy(CUfileBatchHandle_t batch_idp) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  auto it = s.batches.find(static_cast<Batch*>(batch_idp));
  if (it == s.batches.end()) return;
  delete *it;
  s.batches.erase(it);
  if (s.stats_level >= 1) ++s.stats.l1.batch_destroy_ops.ok;
}

// ---- stream-ordered I/O -----------------------------------------------------------
//
// The I/O runs in stream order: after the work queued before it, before the
// work queued after it, reading *size_p and the offsets when it runs. Here it
// runs when the call is made, once the stream has drained -- which is the same
// order -- or, while the stream is being captured, as a host node of the graph
// that reads the parameters each time the graph is launched.

static constexpr unsigned kStreamFlags = CU_FILE_STREAM_FIXED_BUF_OFFSET | CU_FILE_STREAM_FIXED_FILE_OFFSET |
                                         CU_FILE_STREAM_FIXED_FILE_SIZE | CU_FILE_STREAM_PAGE_ALIGNED_INPUTS;

static CUfileError_t io_async(CUfileHandle_t fh, void* buf, size_t* size_p, off_t* file_offset_p,
                              off_t* buf_offset_p, ssize_t* bytes_p, CUstream stream, bool write) {
  if (!fh || !buf || !size_p || !file_offset_p || !buf_offset_p || !bytes_p) return status(CU_FILE_INVALID_VALUE);
  {
    State& s = st();
    std::lock_guard<std::recursive_mutex> lock(s.mu);
    ensure_open(s);
    if (!find_handle(s, fh)) return status(CU_FILE_HANDLE_NOT_REGISTERED);
  }
  auto op = [=] {
    const ssize_t r = io(fh, buf, *size_p, *file_offset_p, *buf_offset_p, write);
    *bytes_p = r < 0 ? -(ssize_t)errno : r;
  };
  cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
  if (vgpu_record_host_op_if_capturing(cs, op)) return kOk;
  if (cudaStreamSynchronize(cs) != cudaSuccess) {
    const cudaError_t e = cudaGetLastError();
    return CUfileError_t{CU_FILE_CUDA_DRIVER_ERROR, static_cast<CUresult>(e)};
  }
  op();
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileReadAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p,
                                            off_t* file_offset_p, off_t* bufPtr_offset_p, ssize_t* bytes_read_p,
                                            CUstream stream) {
  return io_async(fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p, bytes_read_p, stream, false);
}

CUFILE_EXPORT CUfileError_t cuFileWriteAsync(CUfileHandle_t fh, void* bufPtr_base, size_t* size_p,
                                             off_t* file_offset_p, off_t* bufPtr_offset_p,
                                             ssize_t* bytes_written_p, CUstream stream) {
  return io_async(fh, bufPtr_base, size_p, file_offset_p, bufPtr_offset_p, bytes_written_p, stream, true);
}

// Registering a stream twice, or with an unknown flag, is
// CU_FILE_INVALID_VALUE on the card; the default stream registers.
CUFILE_EXPORT CUfileError_t cuFileStreamRegister(CUstream stream, unsigned flags) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  ensure_open(s);
  if (flags & ~kStreamFlags) return status(CU_FILE_INVALID_VALUE);
  if (stream && !s.streams.emplace(stream, flags).second) return status(CU_FILE_INVALID_VALUE);
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileStreamDeregister(CUstream stream) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count == 0) return status(CU_FILE_DRIVER_NOT_INITIALIZED);
  if (stream && !s.streams.erase(stream)) return status(CU_FILE_INVALID_VALUE);
  return kOk;
}

// ---- configuration -----------------------------------------------------------------

CUFILE_EXPORT CUfileError_t cuFileGetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t* value) {
  if (!value || (int)param < 0 || (int)param >= kSizeParams) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count > 0) *value = s.run_size[param];
  else *value = s.staged_size_set[param] ? s.staged_size[param] : kIdleSize[param];
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileGetParameterBool(CUFileBoolConfigParameter_t param, bool* value) {
  if (!value || (int)param < 0 || (int)param >= kBoolParams || !bool_readable(param))
    return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count > 0) *value = s.run_bool[param];
  else *value = s.staged_bool_set[param] ? s.staged_bool[param] : kIdleBool[param];
  return kOk;
}

// The value and its NUL must fit in len bytes: a shorter buffer is
// CU_FILE_INVALID_VALUE on the card, not a truncated copy.
CUFILE_EXPORT CUfileError_t cuFileGetParameterString(CUFileStringConfigParameter_t param, char* desc_str, int len) {
  if (!desc_str || len <= 0 || (int)param < 0 || (int)param >= kStringParams || !string_readable(param))
    return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  const std::string v = s.use_count > 0 ? s.run_string[param]
                                        : (s.staged_string_set[param] ? s.staged_string[param] : std::string());
  if (v.size() >= (size_t)len) return status(CU_FILE_INVALID_VALUE);
  std::memcpy(desc_str, v.data(), v.size());
  desc_str[v.size()] = 0;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileGetParameterMinMaxValue(CUFileSizeTConfigParameter_t param, size_t* min_value,
                                                          size_t* max_value) {
  if (!min_value || !max_value || (int)param <= CUFILE_PARAM_PROFILE_STATS || (int)param >= kSizeParams)
    return status(CU_FILE_INVALID_VALUE);
  *min_value = kMin[param];
  *max_value = kMax[param];
  return kOk;
}

// Setting is for a closed driver only, and a value is not checked until the
// driver opens (the card accepts an io_batchsize of 100000 here, then reports
// it and uses 128).
CUFILE_EXPORT CUfileError_t cuFileSetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t value) {
  if ((int)param < 0 || (int)param >= kSizeParams) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count > 0) return status(CU_FILE_DRIVER_ALREADY_OPEN);
  s.staged_size[param] = value;
  s.staged_size_set[param] = true;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileSetParameterBool(CUFileBoolConfigParameter_t param, bool value) {
  if ((int)param < 0 || (int)param >= kBoolParams) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count > 0) return status(CU_FILE_DRIVER_ALREADY_OPEN);
  s.staged_bool[param] = value;
  s.staged_bool_set[param] = true;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileSetParameterString(CUFileStringConfigParameter_t param, const char* desc_str) {
  if (!desc_str || (int)param < 0 || (int)param >= kStringParams) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count > 0) return status(CU_FILE_DRIVER_ALREADY_OPEN);
  s.staged_string[param] = desc_str;
  s.staged_string_set[param] = true;
  return kOk;
}

// The POSIX bounce-buffer pool. Compatibility mode here stages through one
// buffer, so there is no pool to configure: the setter checks its arguments
// and keeps nothing, and the getter has nothing to report
// (CU_FILE_INVALID_VALUE, which is also the card's answer before an open).
CUFILE_EXPORT CUfileError_t cuFileSetParameterPosixPoolSlabArray(const size_t* size_values,
                                                                 const size_t* count_values, int len) {
  if (!size_values || !count_values || len <= 0) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count > 0) return status(CU_FILE_DRIVER_ALREADY_OPEN);
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileGetParameterPosixPoolSlabArray(size_t* size_values, size_t* count_values,
                                                                 int len) {
  (void)size_values;
  (void)count_values;
  (void)len;
  return status(CU_FILE_INVALID_VALUE);
}

// ---- statistics ---------------------------------------------------------------------

CUFILE_EXPORT CUfileError_t cuFileSetStatsLevel(int level) {
  if (level < 0 || level > 3) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  s.stats_level = level;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileGetStatsLevel(int* level) {
  if (!level) return status(CU_FILE_INVALID_VALUE);
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  *level = s.stats_level;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileStatsStart(void) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count == 0) return status(CU_FILE_DRIVER_NOT_INITIALIZED);
  s.stats_on = true;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileStatsStop(void) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count == 0) return status(CU_FILE_DRIVER_NOT_INITIALIZED);
  s.stats_on = false;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileStatsReset(void) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (s.use_count == 0) return status(CU_FILE_DRIVER_NOT_INITIALIZED);
  s.stats = Stats{};
  return kOk;
}

static CUfileError_t stats_check(State& s, const void* out, int level) {
  if (s.use_count == 0) return status(CU_FILE_DRIVER_NOT_INITIALIZED);
  if (!out || s.stats_level < level) return status(CU_FILE_INVALID_VALUE);
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileGetStatsL1(CUfileStatsLevel1_t* stats) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (CUfileError_t e = stats_check(s, stats, 1); e.err) return e;
  *stats = s.stats.l1;
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileGetStatsL2(CUfileStatsLevel2_t* stats) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (CUfileError_t e = stats_check(s, stats, 2); e.err) return e;
  *stats = CUfileStatsLevel2_t{};
  stats->basic = s.stats.l1;
  std::memcpy(stats->read_size_kb_hist, s.stats.read_hist, sizeof s.stats.read_hist);
  std::memcpy(stats->write_size_kb_hist, s.stats.write_hist, sizeof s.stats.write_hist);
  return kOk;
}

CUFILE_EXPORT CUfileError_t cuFileGetStatsL3(CUfileStatsLevel3_t* stats) {
  State& s = st();
  std::lock_guard<std::recursive_mutex> lock(s.mu);
  if (CUfileError_t e = stats_check(s, stats, 3); e.err) return e;
  *stats = CUfileStatsLevel3_t{};
  stats->detailed.basic = s.stats.l1;
  std::memcpy(stats->detailed.read_size_kb_hist, s.stats.read_hist, sizeof s.stats.read_hist);
  std::memcpy(stats->detailed.write_size_kb_hist, s.stats.write_hist, sizeof s.stats.write_hist);
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();
    n = 0;
  }
  stats->num_gpus = (uint32_t)std::min(n, 16);
  for (uint32_t d = 0; d < stats->num_gpus; ++d) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, (int)d) == cudaSuccess)
      std::memcpy(stats->per_gpu_stats[d].uuid, &prop.uuid, CUFILE_GPU_UUID_LEN);
    else
      cudaGetLastError();
    // Everything goes through the POSIX path in compatibility mode.
    stats->per_gpu_stats[d].read_bytes = d == 0 ? s.stats.l1.read_bytes : 0;
    stats->per_gpu_stats[d].writes_bytes = d == 0 ? s.stats.l1.write_bytes : 0;
    stats->per_gpu_stats[d].n_total_reads = stats->per_gpu_stats[d].n_posix_reads =
        d == 0 ? s.stats.l1.read_ops.ok : 0;
    stats->per_gpu_stats[d].n_total_writes = stats->per_gpu_stats[d].n_posix_writes =
        d == 0 ? s.stats.l1.write_ops.ok : 0;
  }
  return kOk;
}
