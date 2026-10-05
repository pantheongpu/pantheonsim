// NVML: ECC, page retirement, row remapping, accounting, encoder, samples. See nvml_common.inc.
//
// ECC counts, retired pages, remapped rows and PCIe counters come from the
// machine's reliability state (vgpu/ras.hpp), which `vgpu fault` and faults
// delivered to running kernels write, so every number here is zero until
// something is injected. Processes and memory come from the telemetry sample.
// Measurements named below are the RTX 3060's, through NVIDIA's NVML 13.0 (WSL
// driver 596.36); datacenter answers follow the nvml.h documentation.

namespace {

vgpu::ras::State read_ras(const vgpu::telemetry::DeviceSample& d) {
  try {
    return vgpu::ras::read(d.uuid);
  } catch (const std::exception&) {
    return vgpu::ras::State{};
  }
}

unsigned long long now_us() {
  return static_cast<unsigned long long>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

// ---- retired pages ----------------------------------------------------------
//
// The reliability state counts retired pages by cause but does not name them. A
// page needs an address and a time, so the pages are named here, once, when NVML
// first sees them: an address derived from the device's UUID (a 4 KiB-aligned
// address inside its memory, different for every page and stable for the life of
// the card) and the time they were first seen, kept in the device's persistent
// settings so a later process lists the same pages. The time is therefore when
// NVML noticed the retirement, not when it happened: the simulator does not
// timestamp an injection per page.
struct RetiredPage {
  unsigned long long address, timestamp;
  int cause;   // 0 multiple single-bit errors, 1 double-bit error
};

std::vector<RetiredPage> retired_pages(const vgpu::telemetry::DeviceSample& d) {
  std::vector<RetiredPage> out;
  const vgpu::ras::State st = read_ras(d);
  const uint64_t want[2] = {st.lifetime.retired_sbe, st.lifetime.retired_dbe};
  const std::string text = setting(d.uuid, Life::Persistent, "retired_pages", "");
  std::istringstream in(text);
  for (std::string tok; std::getline(in, tok, ';');) {
    unsigned long long a = 0, t = 0;
    int c = 0;
    if (std::sscanf(tok.c_str(), "%llx:%d:%llu", &a, &c, &t) == 3) out.push_back({a, t, c});
  }
  uint64_t have[2] = {0, 0};
  for (const RetiredPage& p : out) ++have[p.cause & 1];
  bool grew = false;
  unsigned long long h = 1469598103934665603ull;
  for (const char* c = d.uuid; *c; ++c) h = (h ^ static_cast<unsigned char>(*c)) * 1099511628211ull;
  const unsigned long long pages = d.vram_total_bytes >= 4096 ? d.vram_total_bytes / 4096 : 1;
  for (int cause = 0; cause < 2; ++cause)
    while (have[cause] < want[cause] && out.size() < 4096) {
      const unsigned long long k = out.size() + 1;
      out.push_back({((h + k * 0x9e3779b97f4a7c15ull) % pages) * 4096, now_us(), cause});
      ++have[cause];
      grew = true;
    }
  if (grew) {
    std::string s;
    char buf[80];
    for (const RetiredPage& p : out) {
      std::snprintf(buf, sizeof buf, "%llx:%d:%llu;", p.address, p.cause, p.timestamp);
      s += buf;
    }
    overlay_set(d.uuid, Life::Persistent, "retired_pages", s);
  }
  return out;
}

nvmlReturn_t retired_pages_common(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount,
                                  unsigned long long* addresses, unsigned long long* timestamps) {
  NVML_DEV(device);
  NVML_ARG(pageCount);
  const int c = static_cast<int>(cause);
  if (c < 0 || c >= 2) return NVML_ERROR_INVALID_ARGUMENT;
  // Page retirement is what GDDR cards with ECC do; no ECC (the card: NOT_SUPPORTED)
  // and HBM (which remaps rows) have none.
  if (d->memory_retirement != 1) return NVML_ERROR_NOT_SUPPORTED;
  std::vector<RetiredPage> mine;
  for (const RetiredPage& p : retired_pages(*d))
    if (p.cause == c) mine.push_back(p);
  const unsigned have = static_cast<unsigned>(mine.size());
  if (*pageCount < have) {
    *pageCount = have;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  if (have && !addresses) return NVML_ERROR_INVALID_ARGUMENT;
  for (unsigned i = 0; i < have; ++i) {
    addresses[i] = mine[i].address;
    if (timestamps) timestamps[i] = mine[i].timestamp;
  }
  *pageCount = have;
  return NVML_SUCCESS;
}

}  // namespace

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause,
                                                   unsigned int* pageCount, unsigned long long* addresses) {
  return retired_pages_common(device, cause, pageCount, addresses, nullptr);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause,
                                                      unsigned int* pageCount, unsigned long long* addresses,
                                                      unsigned long long* timestamps) {
  return retired_pages_common(device, cause, pageCount, addresses, timestamps);
}
// Pending until the next driver load completes the retirement (vgpu/ras.hpp).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t* isPending) {
  NVML_DEV(device);
  NVML_ARG(isPending);
  if (d->memory_retirement != 1) return NVML_ERROR_NOT_SUPPORTED;
  *isPending = read_ras(*d).lifetime.retired_pending ? NVML_FEATURE_ENABLED : NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}

/* ---- row remapping (HBM cards) ---- */

VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrRows,
                                                   unsigned int* uncRows, unsigned int* isPending,
                                                   unsigned int* failureOccurred) {
  NVML_DEV(device);
  if (!corrRows || !uncRows || !isPending || !failureOccurred) return NVML_ERROR_INVALID_ARGUMENT;
  if (d->memory_retirement != 2) return NVML_ERROR_NOT_SUPPORTED;   // the card (GDDR, no ECC): NOT_SUPPORTED
  const vgpu::ras::Counters c = read_ras(*d).lifetime;
  *corrRows = static_cast<unsigned>(c.rows_correctable);
  *uncRows = static_cast<unsigned>(c.rows_uncorrectable);
  *isPending = c.rows_pending ? 1 : 0;
  *failureOccurred = c.rows_failure ? 1 : 0;
  return NVML_SUCCESS;
}
// How many banks have each degree of remapping headroom. A part's bank count is
// not in any datasheet this project uses, so there is no histogram to give.
VGPU_NVML_NS_DEV(nvmlDeviceGetRowRemapperHistogram, device, (nvmlDevice_t device, nvmlRowRemapperHistogramValues_t*))
#if defined(VGPU_NVML_13_0)
// TPC and channel repair: the card (Ampere) answers nothing pending, and nothing
// here ever needs repair. Turing has no such repair.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRepairStatus(nvmlDevice_t device, nvmlRepairStatus_t* status) {
  NVML_DEV(device);
  NVML_ARG(status);
  NVML_VERSION(status, nvmlRepairStatus_v1);
  if (d->cc_major < 8) return NVML_ERROR_NOT_SUPPORTED;
  status->bChannelRepairPending = 0;
  status->bTpcRepairPending = 0;
  return NVML_SUCCESS;
}
#endif
#if defined(VGPU_NVML_13_2)
// Hopper and newer: set once a row remap has failed, which is memory the GPU can
// no longer repair.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetUnrepairableMemoryFlag_v1(nvmlDevice_t device,
                                                                nvmlUnrepairableMemoryStatus_v1_t* status) {
  NVML_DEV(device);
  NVML_ARG(status);
  if (d->cc_major < 9) return NVML_ERROR_NOT_SUPPORTED;
  status->bUnrepairableMemory = read_ras(*d).lifetime.rows_failure ? 1 : 0;
  return NVML_SUCCESS;
}
#endif

/* ---- ECC ---- */

// The mode a card ships in (the profile's: on for a card with ECC); only a card
// with ECC has one.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* defaultMode) {
  NVML_DEV(device);
  NVML_ARG(defaultMode);
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  *defaultMode = NVML_FEATURE_ENABLED;
  return NVML_SUCCESS;
}
// The deprecated by-location counts: L1, L2, device memory, register file.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType,
                                                        nvmlEccCounterType_t counterType,
                                                        nvmlEccErrorCounts_t* eccCounts) {
  NVML_DEV(device);
  const int et = static_cast<int>(errorType), ct = static_cast<int>(counterType);
  if (!eccCounts || et < 0 || et > 1 || ct < 0 || ct > 1) return NVML_ERROR_INVALID_ARGUMENT;
  if (!d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  const vgpu::ras::State st = read_ras(*d);
  const vgpu::ras::Counters& c = ct == 1 ? st.lifetime : st.since_load;
  using L = vgpu::ras::Location;
  eccCounts->l1Cache = c.ecc[et][static_cast<int>(L::L1Cache)];
  eccCounts->l2Cache = c.ecc[et][static_cast<int>(L::L2Cache)];
  eccCounts->deviceMemory = c.ecc[et][static_cast<int>(L::DeviceMemory)];
  eccCounts->registerFile = c.ecc[et][static_cast<int>(L::RegisterFile)];
  return NVML_SUCCESS;
}
#if defined(VGPU_NVML_12_4)
// SRAM ECC error status (Ampere and newer): the counts of the on-chip memories,
// aggregate and since the driver loaded. The card answers all zeros, a GeForce
// with no ECC included. Detected single-bit errors are the corrected count and
// double-bit errors the uncorrectable SEC-DED one; the per-unit buckets are not
// tracked and stay zero.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSramEccErrorStatus(nvmlDevice_t device, nvmlEccSramErrorStatus_t* status) {
  NVML_DEV(device);
  NVML_ARG(status);
  NVML_VERSION(status, nvmlEccSramErrorStatus_v1);
  if (d->cc_major < 8) return NVML_ERROR_NOT_SUPPORTED;
  const vgpu::ras::State st = read_ras(*d);
  const int sram = static_cast<int>(vgpu::ras::Location::Sram);
  std::memset(status, 0, sizeof *status);
  status->version = nvmlEccSramErrorStatus_v1;
  status->aggregateCor = st.lifetime.ecc[0][sram];
  status->aggregateUncSecDed = st.lifetime.ecc[1][sram];
  status->volatileCor = st.since_load.ecc[0][sram];
  status->volatileUncSecDed = st.since_load.ecc[1][sram];
  return NVML_SUCCESS;
}
#endif
#if defined(VGPU_NVML_13_0)
// Blackwell only, and only with ECC on; the unique errors are not tracked, so
// the list is empty.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSramUniqueUncorrectedEccErrorCounts(
    nvmlDevice_t device, nvmlEccSramUniqueUncorrectedErrorCounts_t* counts) {
  NVML_DEV(device);
  NVML_ARG(counts);
  NVML_VERSION(counts, nvmlEccSramUniqueUncorrectedErrorCounts_v1);
  if (d->cc_major < 10 || !d->ecc_enabled) return NVML_ERROR_NOT_SUPPORTED;
  counts->entryCount = 0;
  return NVML_SUCCESS;
}
#endif

/* ---- processes ---- */

namespace {
// The v1 process records: the process id and its memory, nothing else.
nvmlReturn_t processes_v1(nvmlDevice_t device, bool compute, unsigned int* infoCount, nvmlProcessInfo_v1_t* infos) {
  NVML_DEV(device);
  NVML_ARG(infoCount);
  const unsigned have = compute ? d->proc_count : 0;
  if (!infos || *infoCount < have) {
    *infoCount = have;
    return have ? NVML_ERROR_INSUFFICIENT_SIZE : NVML_SUCCESS;
  }
  for (unsigned i = 0; i < have; ++i) {
    infos[i].pid = d->procs[i].pid;
    infos[i].usedGpuMemory = d->procs[i].used_bytes;
  }
  *infoCount = have;
  return NVML_SUCCESS;
}
}  // namespace
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int* infoCount,
                                                              nvmlProcessInfo_v1_t* infos) {
  return processes_v1(device, true, infoCount, infos);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t device, unsigned int* infoCount,
                                                               nvmlProcessInfo_v1_t* infos) {
  return processes_v1(device, false, infoCount, infos);
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses(nvmlDevice_t device, unsigned int* infoCount,
                                                                 nvmlProcessInfo_v1_t* infos) {
  return processes_v1(device, false, infoCount, infos);
}
#if defined(VGPU_NVML_12_4)
// Process details (Hopper and newer; the card, Ampere, says NOT_SUPPORTED): the
// compute processes (mode 0) with their memory, no instance ids, no protected
// memory; graphics (1) and MPS (2) have none.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetRunningProcessDetailList(nvmlDevice_t device, nvmlProcessDetailList_t* plist) {
  NVML_DEV(device);
  NVML_ARG(plist);
  NVML_VERSION(plist, nvmlProcessDetailList_v1);
  if (plist->mode > 2) return NVML_ERROR_INVALID_ARGUMENT;
  if (d->cc_major < 9) return NVML_ERROR_NOT_SUPPORTED;
  const unsigned have = plist->mode == 0 ? d->proc_count : 0;
  if (!plist->procArray || plist->numProcArrayEntries < have) {
    plist->numProcArrayEntries = have;
    return have ? NVML_ERROR_INSUFFICIENT_SIZE : NVML_SUCCESS;
  }
  for (unsigned i = 0; i < have; ++i) {
    nvmlProcessDetail_v1_t& p = plist->procArray[i];
    std::memset(&p, 0, sizeof p);
    p.pid = d->procs[i].pid;
    p.usedGpuMemory = d->procs[i].used_bytes;
    p.gpuInstanceId = 0xFFFFFFFFu;
    p.computeInstanceId = 0xFFFFFFFFu;
  }
  plist->numProcArrayEntries = have;
  return NVML_SUCCESS;
}
// Per-process utilization: VirtualGPU measures the device's utilization and has no
// per-process share of it, so there are never samples to return -- the answer
// nvmlDeviceGetProcessUtilization gives. The card answers INSUFFICIENT_SIZE for
// the NULL array the caller sizes with first.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetProcessesUtilizationInfo(nvmlDevice_t device,
                                                               nvmlProcessesUtilizationInfo_t* info) {
  NVML_DEV(device);
  NVML_ARG(info);
  NVML_VERSION(info, nvmlProcessesUtilizationInfo_v1);
  info->processSamplesCount = 0;
  return info->procUtilArray ? NVML_ERROR_NOT_FOUND : NVML_ERROR_INSUFFICIENT_SIZE;
}
#endif

/* ---- accounting ----
 *
 * Accounting mode (nvmlDeviceSetAccountingMode) makes the driver keep a record of
 * each process that used the GPU, running or finished. The record is built from
 * what the telemetry shows, every time NVML is asked about accounting while it
 * is on: a process seen for the first time starts a record; its memory high-water
 * mark follows what the telemetry reports of it; a process that is gone from the
 * telemetry is finished, its time the span between the first and the last time it
 * was seen. A process that starts and ends between two observations is never
 * seen. Utilization is not attributed to processes in the model, so it reads
 * NVML_VALUE_NOT_AVAILABLE, which the struct documents for a driver that cannot
 * say. The mode and the records live with the session (volatile), as a driver
 * reload would reset them.
 */
namespace {

struct AcctRecord {
  unsigned pid;
  unsigned long long start_us, max_mem, last_us;
  bool running;
};

std::string acct_path(const std::string& uuid) {
  return vgpu::telemetry::default_path() + "/nvml-acct-" + uuid + ".dat";
}

bool accounting_on(const vgpu::telemetry::DeviceSample& d) {
  return setting_int(d.uuid, Life::Volatile, "accounting", 0) != 0;
}

// Brings the records up to date with the processes the telemetry shows now, and
// returns them in the order they started (oldest first). `clear` 1 forgets every
// record, 2 the finished ones.
std::vector<AcctRecord> acct_sync(const vgpu::telemetry::DeviceSample& d, int clear = 0) {
  std::vector<AcctRecord> recs;
  const std::string path = acct_path(d.uuid);
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd < 0) return recs;
  ::flock(fd, LOCK_EX);
  std::string text;
  {
    char buf[4096];
    ssize_t n;
    off_t off = 0;
    while ((n = ::pread(fd, buf, sizeof buf, off)) > 0) {
      text.append(buf, static_cast<size_t>(n));
      off += n;
    }
  }
  {
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
      AcctRecord r{};
      int running = 0;
      if (std::sscanf(line.c_str(), "%u %llu %llu %llu %d", &r.pid, &r.start_us, &r.max_mem, &r.last_us, &running) == 5) {
        r.running = running != 0;
        recs.push_back(r);
      }
    }
    for (AcctRecord& r : recs) {
      bool alive = false;
      for (unsigned i = 0; i < d.proc_count; ++i)
        if (d.procs[i].pid == r.pid) alive = true;
      if (r.running && !alive) r.running = false;   // finished: its last sighting stands as its end
    }
    const unsigned long long now = now_us();
    for (unsigned i = 0; i < d.proc_count; ++i) {
      AcctRecord* mine = nullptr;
      for (AcctRecord& r : recs)
        if (r.pid == d.procs[i].pid && r.running) mine = &r;
      if (!mine) {
        recs.push_back({d.procs[i].pid, now, 0, now, true});
        mine = &recs.back();
      }
      mine->max_mem = std::max<unsigned long long>(mine->max_mem, d.procs[i].used_bytes);
      mine->last_us = now;
    }
    if (clear == 1) recs.clear();
    if (clear == 2) recs.erase(std::remove_if(recs.begin(), recs.end(), [](const AcctRecord& r) { return !r.running; }), recs.end());
    // The driver's buffer is circular: the newest records stay (the buffer size
    // is what nvmlDeviceGetAccountingBufferSize reports).
    if (recs.size() > 4000) recs.erase(recs.begin(), recs.begin() + static_cast<long>(recs.size() - 4000));
  }
  std::string out;
  char buf[128];
  for (const AcctRecord& r : recs) {
    std::snprintf(buf, sizeof buf, "%u %llu %llu %llu %d\n", r.pid, r.start_us, r.max_mem, r.last_us, r.running ? 1 : 0);
    out += buf;
  }
  if (::ftruncate(fd, 0) == 0) {
    size_t done = 0;
    while (done < out.size()) {
      const ssize_t n = ::pwrite(fd, out.data() + done, out.size() - done, static_cast<off_t>(done));
      if (n <= 0) break;
      done += static_cast<size_t>(n);
    }
  }
  ::flock(fd, LOCK_UN);
  ::close(fd);
  return recs;
}

}  // namespace

// The driver keeps 4000 records (the card answers 4000).
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* bufferSize) {
  NVML_DEV(device);
  NVML_ARG(bufferSize);
  *bufferSize = 4000;
  return NVML_SUCCESS;
}
// Off by default, as on the card.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
  NVML_DEV(device);
  NVML_ARG(mode);
  *mode = accounting_on(*d) ? NVML_FEATURE_ENABLED : NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count, unsigned int* pids) {
  NVML_DEV(device);
  NVML_ARG(count);
  // Measured: with accounting off the card answers an empty list, not
  // NOT_SUPPORTED as nvml.h says.
  const std::vector<AcctRecord> recs = accounting_on(*d) ? acct_sync(*d) : std::vector<AcctRecord>{};
  const unsigned have = static_cast<unsigned>(recs.size());
  // Measured: the sizing call (a NULL list) is INSUFFICIENT_SIZE with the count,
  // as nvml.h documents, and so is any list the records do not fit.
  if (!pids || *count < have) {
    *count = have;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  for (unsigned i = 0; i < have; ++i) pids[i] = recs[i].pid;
  *count = have;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid,
                                                      nvmlAccountingStats_t* stats) {
  NVML_DEV(device);
  NVML_ARG(stats);
  // Measured: with accounting off no pid is found.
  if (!accounting_on(*d)) return NVML_ERROR_NOT_FOUND;
  const std::vector<AcctRecord> recs = acct_sync(*d);
  for (auto it = recs.rbegin(); it != recs.rend(); ++it) {   // a reused pid reports its latest process
    if (it->pid != pid) continue;
    std::memset(stats, 0, sizeof *stats);
    stats->gpuUtilization = static_cast<unsigned>(NVML_VALUE_NOT_AVAILABLE);
    stats->memoryUtilization = static_cast<unsigned>(NVML_VALUE_NOT_AVAILABLE);
    stats->maxMemoryUsage = it->max_mem;
    stats->startTime = it->start_us;
    stats->isRunning = it->running ? 1 : 0;
    stats->time = it->running ? 0 : (it->last_us - it->start_us) / 1000;
    return NVML_SUCCESS;
  }
  return NVML_ERROR_NOT_FOUND;   // measured: a pid the GPU never saw
}
// Used by the setters (nvml_set.inc): forget every record.
namespace {
void accounting_clear(const vgpu::telemetry::DeviceSample& d, bool finished_only) {
  acct_sync(d, finished_only ? 2 : 1);
}
}  // namespace

/* ---- video engines, frame buffer capture, utilization ---- */

namespace {
// Engines present, from the profile: NVENC and NVDEC; the JPEG engine arrived
// with Ampere and the optical flow accelerator with Turing.
bool has_encoder(const vgpu::telemetry::DeviceSample& d) { return nvml_facts(d).nvenc; }
bool has_decoder(const vgpu::telemetry::DeviceSample& d) { return nvml_facts(d).nvdec; }
bool has_jpeg(const vgpu::telemetry::DeviceSample& d) { return d.cc_major >= 8; }
bool has_ofa(const vgpu::telemetry::DeviceSample& d) {
  return d.cc_major > 7 || (d.cc_major == 7 && d.cc_minor >= 5);
}
}  // namespace

// The sampling period of the engine utilization queries: 200 ms, as measured.
// An engine nobody drives reads 0; none of the simulated programs encodes. The
// card answers 0 for all four engines; this used to be NOT_SUPPORTED for the
// encoder and decoder, which the card shows was wrong.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization,
                                                         unsigned int* samplingPeriodUs) {
  NVML_DEV(device);
  if (!utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;
  if (!has_encoder(*d)) return NVML_ERROR_NOT_SUPPORTED;
  *utilization = 0;
  *samplingPeriodUs = 200000;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization,
                                                         unsigned int* samplingPeriodUs) {
  NVML_DEV(device);
  if (!utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;
  if (!has_decoder(*d)) return NVML_ERROR_NOT_SUPPORTED;
  *utilization = 0;
  *samplingPeriodUs = 200000;
  return NVML_SUCCESS;
}
#if defined(VGPU_NVML_12_4)
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetJpgUtilization(nvmlDevice_t device, unsigned int* utilization,
                                                     unsigned int* samplingPeriodUs) {
  NVML_DEV(device);
  if (!utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;
  if (!has_jpeg(*d)) return NVML_ERROR_NOT_SUPPORTED;
  *utilization = 0;
  *samplingPeriodUs = 200000;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetOfaUtilization(nvmlDevice_t device, unsigned int* utilization,
                                                     unsigned int* samplingPeriodUs) {
  NVML_DEV(device);
  if (!utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;
  if (!has_ofa(*d)) return NVML_ERROR_NOT_SUPPORTED;
  *utilization = 0;
  *samplingPeriodUs = 200000;
  return NVML_SUCCESS;
}
#endif
// How many more encode sessions the GPU takes, in percent of one engine: 100 for a
// free codec. Measured on the card: H264 and HEVC 100, AV1 NOT_SUPPORTED (Ada and
// newer have AV1 encoders), any other type INVALID_ARGUMENT.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType,
                                                      unsigned int* encoderCapacity) {
  NVML_DEV(device);
  NVML_ARG(encoderCapacity);
  const int t = static_cast<int>(encoderQueryType);
  if (t < 0 || t > 2) return NVML_ERROR_INVALID_ARGUMENT;
  if (!has_encoder(*d) || (t == 2 && !(d->cc_major > 8 || (d->cc_major == 8 && d->cc_minor >= 9))))
    return NVML_ERROR_NOT_SUPPORTED;
  *encoderCapacity = 100;
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount,
                                                   unsigned int* averageFps, unsigned int* averageLatency) {
  NVML_DEV(device);
  if (!sessionCount || !averageFps || !averageLatency) return NVML_ERROR_INVALID_ARGUMENT;
  if (!has_encoder(*d)) return NVML_ERROR_NOT_SUPPORTED;
  *sessionCount = *averageFps = *averageLatency = 0;
  return NVML_SUCCESS;
}
// No program encodes here: no sessions, as the card (nothing encoding) answers.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int* sessionCount,
                                                      nvmlEncoderSessionInfo_t* sessionInfos) {
  NVML_DEV(device);
  NVML_ARG(sessionCount);
  (void)sessionInfos;
  if (!has_encoder(*d)) return NVML_ERROR_NOT_SUPPORTED;
  *sessionCount = 0;
  return NVML_SUCCESS;
}
// NvFBC (frame buffer capture) is a GeForce and workstation feature: nothing
// captures here (sessions and statistics are zero, as on the card); a datacenter
// board does not offer it.
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t* fbcStats) {
  NVML_DEV(device);
  NVML_ARG(fbcStats);
  if (!consumer(*d)) return NVML_ERROR_NOT_SUPPORTED;
  std::memset(fbcStats, 0, sizeof *fbcStats);
  return NVML_SUCCESS;
}
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int* sessionCount,
                                                  nvmlFBCSessionInfo_t* sessionInfo) {
  NVML_DEV(device);
  NVML_ARG(sessionCount);
  (void)sessionInfo;
  if (!consumer(*d)) return NVML_ERROR_NOT_SUPPORTED;
  *sessionCount = 0;
  return NVML_SUCCESS;
}

/* ---- samples ----
 *
 * The driver keeps a buffer of recent readings at a high rate. The telemetry
 * keeps the latest reading only, so a query finds one sample, the current value,
 * stamped now, and none if the caller has already seen a sample from this
 * microsecond.
 */
VGPU_EXPORT nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type,
                                              unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType,
                                              unsigned int* sampleCount, nvmlSample_t* samples) {
  NVML_DEV(device);
  if (!sampleValType || !sampleCount) return NVML_ERROR_INVALID_ARGUMENT;
  const int t = static_cast<int>(type);
  if (t < 0 || t >= 10) return NVML_ERROR_INVALID_ARGUMENT;
  if (samples && *sampleCount == 0) return NVML_ERROR_INVALID_ARGUMENT;
  unsigned value = 0;
  switch (t) {
    case 0: value = d->power_mw; break;                                                // total power
    case 1: value = d->utilization_gpu; break;
    case 2: value = d->utilization_mem; break;
    case 3: if (!has_encoder(*d)) return NVML_ERROR_NOT_SUPPORTED; break;
    case 4: if (!has_decoder(*d)) return NVML_ERROR_NOT_SUPPORTED; break;
    case 5: value = d->sm_clock_mhz; break;                                            // processor clock
    case 6: value = d->mem_clock_mhz; break;
    case 7: return NVML_ERROR_NOT_SUPPORTED;                                           // a Grace Hopper module's power
    case 8: if (!has_jpeg(*d)) return NVML_ERROR_NOT_SUPPORTED; break;
    default: if (!has_ofa(*d)) return NVML_ERROR_NOT_SUPPORTED; break;
  }
  const unsigned long long now = now_us();
  *sampleValType = NVML_VALUE_TYPE_UNSIGNED_INT;
  if (lastSeenTimeStamp >= now) {
    *sampleCount = 0;
    return NVML_ERROR_NOT_FOUND;
  }
  if (samples) {
    samples[0].timeStamp = now;
    samples[0].sampleValue.uiVal = value;
  }
  *sampleCount = 1;
  return NVML_SUCCESS;
}
