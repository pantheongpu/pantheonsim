// HSA's shared virtual memory (SVM): attributes on ranges of the process's
// own memory, and prefetches. Included by hsa_api.cpp.
//
// The runtime reports SVM support, as ROCm's does on a machine with HMM, so
// ROCm's HIP over it keeps managed memory the way it does on such a machine:
// plain host pages, made accessible to each GPU (AGENT_ACCESSIBLE), advised
// and prefetched through these calls. Making a range accessible to a GPU maps
// it for the devices where it is (map_host's identity mapping), as locking
// memory does; everything else is kept, range by range, for
// hsa_amd_svm_attributes_get to read back -- there are no pages to migrate.

namespace {

struct SvmAttributes {
  uint64_t preferred = 0;     // an agent's handle, or 0 for none
  uint64_t prefetched = 0;    // where the last prefetch went, or 0
  bool read_mostly = false, read_only = false;
  uint32_t global_flag = 0;   // HSA_AMD_SVM_GLOBAL_FLAG_*: fine-grained unless set
  std::vector<uint64_t> accessible, in_place;   // agents given access
};
// Non-overlapping ranges, by start: [start, end) and what they were given.
std::map<uint64_t, std::pair<uint64_t, SvmAttributes>> g_svm;   // under g_mutex
std::map<uint64_t, uint64_t> g_svm_mapped;                       // host ranges mapped for the devices, under g_mutex

enum : uint64_t {
  kSvmGlobalFlag = 0, kSvmReadOnly = 1, kSvmHiveLocal = 2, kSvmGranularity = 3, kSvmPreferred = 4,
  kSvmPrefetch = 5, kSvmReadMostly = 6, kSvmGpuExec = 7, kSvmAccessible = 0x200, kSvmAccessibleInPlace = 0x201,
  kSvmNoAccess = 0x202, kSvmAccessQuery = 0x203,
};
constexpr uint32_t kSvmIndeterminate = 2;
constexpr uint64_t kSvmPage = 4096;

// A range as the kernel's driver keeps it: whole pages.
std::pair<uint64_t, uint64_t> svm_pages(const void* ptr, size_t size) {
  const uint64_t at = reinterpret_cast<uint64_t>(ptr);
  return {at & ~(kSvmPage - 1), (at + size + kSvmPage - 1) & ~(kSvmPage - 1)};
}

// Splits the records so that one starts at `at`. Under g_mutex.
void svm_split(uint64_t at) {
  auto it = g_svm.upper_bound(at);
  if (it == g_svm.begin()) return;
  --it;
  const uint64_t start = it->first, end = it->second.first;
  if (start < at && at < end) {
    SvmAttributes a = it->second.second;
    it->second.first = at;
    g_svm.emplace(at, std::make_pair(end, std::move(a)));
  }
}
// Every byte of [lo, hi) covered by a record, the gaps given fresh ones,
// and `fn` applied to each. Under g_mutex.
extern "C++" template <typename Fn>
void svm_cover(uint64_t lo, uint64_t hi, Fn fn) {
  svm_split(lo);
  svm_split(hi);
  uint64_t at = lo;
  auto it = g_svm.lower_bound(lo);
  while (at < hi) {
    if (it == g_svm.end() || it->first > at) {
      const uint64_t gap_end = it == g_svm.end() ? hi : std::min(hi, it->first);
      it = g_svm.emplace_hint(it, at, std::make_pair(gap_end, SvmAttributes{}));
    }
    fn(it->second.second);
    at = it->second.first;
    ++it;
  }
}
void add_agent(std::vector<uint64_t>& v, uint64_t a) {
  if (std::find(v.begin(), v.end(), a) == v.end()) v.push_back(a);
}
void drop_agent(std::vector<uint64_t>& v, uint64_t a) { v.erase(std::remove(v.begin(), v.end(), a), v.end()); }

// The devices reach a range of host memory where it is, once.
void svm_map_for_devices(uint64_t lo, uint64_t hi) {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_svm_mapped.upper_bound(lo);
    if (it != g_svm_mapped.begin() && std::prev(it)->second >= hi) return;   // already
    g_svm_mapped[lo] = std::max(hi, g_svm_mapped[lo]);
  }
  shared::map_host(reinterpret_cast<void*>(lo), hi - lo);
}
}  // namespace

hsa_status_t hsa_amd_svm_attributes_set(void* ptr, size_t size, hsa_amd_svm_attribute_pair_t* list, size_t count) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr || !size || (count && !list)) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const auto [lo, hi] = svm_pages(ptr, size);
  bool map = false;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t attr = list[i].attribute, v = list[i].value;
    switch (attr) {
      case kSvmGlobalFlag:
      case kSvmReadOnly:
      case kSvmHiveLocal:
      case kSvmGranularity:
      case kSvmReadMostly:
      case kSvmGpuExec:
        break;
      case kSvmPreferred:
        if (v && !valid_agent(hsa_agent_t{v})) return HSA_STATUS_ERROR_INVALID_AGENT;
        break;
      case kSvmAccessible:
      case kSvmAccessibleInPlace:
      case kSvmNoAccess:
        if (!valid_agent(hsa_agent_t{v})) return HSA_STATUS_ERROR_INVALID_AGENT;
        if (attr != kSvmNoAccess && gpu_of(hsa_agent_t{v}) >= 0) map = true;
        break;
      default:   // PREFETCH_LOCATION and ACCESS_QUERY are only asked about
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    svm_cover(lo, hi, [&](SvmAttributes& a) {
      for (size_t i = 0; i < count; ++i) {
        const uint64_t v = list[i].value;
        switch (list[i].attribute) {
          case kSvmGlobalFlag: a.global_flag = static_cast<uint32_t>(v); break;
          case kSvmReadOnly: a.read_only = v != 0; break;
          case kSvmReadMostly: a.read_mostly = v != 0; break;
          case kSvmPreferred: a.preferred = v; break;
          case kSvmAccessible:
            add_agent(a.accessible, v);
            drop_agent(a.in_place, v);
            break;
          case kSvmAccessibleInPlace:
            add_agent(a.in_place, v);
            drop_agent(a.accessible, v);
            break;
          case kSvmNoAccess:
            drop_agent(a.accessible, v);
            drop_agent(a.in_place, v);
            break;
          default: break;
        }
      }
    });
  }
  // Host memory a GPU is given is mapped for the devices; device memory is
  // theirs already.
  if (map && shared::owner(lo) < 0) svm_map_for_devices(lo, hi);
  return HSA_STATUS_SUCCESS;
}

// Each attribute asked about, across the range: its value where every byte
// agrees, or the "not uniform" answer (no agent, the indeterminate flag).
hsa_status_t hsa_amd_svm_attributes_get(void* ptr, size_t size, hsa_amd_svm_attribute_pair_t* list, size_t count) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr || !size || (count && !list)) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const auto [lo, hi] = svm_pages(ptr, size);
  std::vector<SvmAttributes> parts;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    uint64_t at = lo;
    auto it = g_svm.upper_bound(lo);
    if (it != g_svm.begin()) --it;
    for (; at < hi; ++it) {
      if (it == g_svm.end() || it->first >= hi) {
        parts.emplace_back();   // the rest, never given anything
        break;
      }
      if (it->second.first <= at) continue;
      if (it->first > at) parts.emplace_back();   // a gap before it
      parts.push_back(it->second.second);
      at = it->second.first;
    }
  }
  const auto uniform = [&](auto get, uint64_t mixed) {
    const uint64_t first = get(parts.front());
    for (const SvmAttributes& p : parts)
      if (get(p) != first) return mixed;
    return first;
  };
  for (size_t i = 0; i < count; ++i) {
    uint64_t& v = list[i].value;
    switch (list[i].attribute) {
      case kSvmGlobalFlag: v = uniform([](const SvmAttributes& a) { return uint64_t{a.global_flag}; }, kSvmIndeterminate); break;
      case kSvmReadOnly: v = uniform([](const SvmAttributes& a) { return uint64_t{a.read_only}; }, 0); break;
      case kSvmReadMostly: v = uniform([](const SvmAttributes& a) { return uint64_t{a.read_mostly}; }, 0); break;
      case kSvmPreferred: v = uniform([](const SvmAttributes& a) { return a.preferred; }, 0); break;
      case kSvmPrefetch: v = uniform([](const SvmAttributes& a) { return a.prefetched; }, 0); break;
      case kSvmHiveLocal: v = 0; break;
      case kSvmGranularity: v = 9; break;   // 2^9 pages, ROCm's default
      case kSvmGpuExec: v = 1; break;
      case kSvmAccessQuery: {
        // The agent named, as the whole range has it.
        const uint64_t agent = v;
        if (!valid_agent(hsa_agent_t{agent})) return HSA_STATUS_ERROR_INVALID_AGENT;
        v = uniform(
            [agent](const SvmAttributes& a) {
              if (std::find(a.in_place.begin(), a.in_place.end(), agent) != a.in_place.end()) return kSvmAccessibleInPlace;
              if (std::find(a.accessible.begin(), a.accessible.end(), agent) != a.accessible.end()) return kSvmAccessible;
              return kSvmNoAccess;
            },
            kSvmNoAccess);
        list[i].attribute = v;
        v = agent;
        break;
      }
      default: return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }
  return HSA_STATUS_SUCCESS;
}

// A prefetch has nothing to move: it waits for what it depends on, records
// where it went, and signals that it is done.
hsa_status_t hsa_amd_svm_prefetch_async(void* ptr, size_t size, hsa_agent_t agent, uint32_t num_deps,
                                        const hsa_signal_t* deps, hsa_signal_t completion) {
  if (!started()) return HSA_STATUS_ERROR_NOT_INITIALIZED;
  if (!ptr || !size || (num_deps && !deps)) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (!valid_agent(agent)) return HSA_STATUS_ERROR_INVALID_AGENT;
  for (uint32_t i = 0; i < num_deps; ++i)
    hsa_signal_wait_scacquire(deps[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  const auto [lo, hi] = svm_pages(ptr, size);
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    svm_cover(lo, hi, [&](SvmAttributes& a) { a.prefetched = agent.handle; });
  }
  if (completion.handle) hsa_signal_subtract_screlease(completion, 1);
  return HSA_STATUS_SUCCESS;
}
