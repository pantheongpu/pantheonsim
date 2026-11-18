// Launch-level result caching (vgpu/memo.hpp, vgpu/runtime/memo.hpp). See docs/perf-design/rounds/10-rounds-sp19-sp27.md, "Round sp35".
#include "vgpu/runtime/memo.hpp"

#include <dirent.h>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <unordered_set>
#include <variant>

#include "vgpu/profiling.hpp"
#include "vgpu/ptx/ast.hpp"
#include "vgpu/sass/exec.hpp"

extern char** environ;

namespace vgpu::memo {

namespace {

inline uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline bool is_device_ptr(uint64_t v) { return v >= kDeviceVaBase && v < kDeviceVaEnd; }

// Whether any aligned word of the `n` bytes at device address `addr` could be a device address.
bool has_pointer_like(uint64_t addr, const uint8_t* p, uint64_t n) {
  uint64_t i = (4 - addr % 4) % 4;   // every 4-aligned word: a pointer in a packed struct or stored as two halves' neighbours
  for (; i + 8 <= n; i += 4) {
    uint64_t w;
    std::memcpy(&w, p + i, 8);
    if (is_device_ptr(w)) return true;
  }
  return false;
}

std::string hex128(const Hash128& h) {
  char b[40];
  std::snprintf(b, sizeof b, "%016" PRIx64 "%016" PRIx64, h.a, h.b);
  return b;
}

uint64_t env_u64(const char* name, uint64_t dflt) {
  const char* e = std::getenv(name);
  if (!e || !*e) return dflt;
  return std::strtoull(e, nullptr, 10);
}

// The identity of the code this runs. Preferably the object's own GNU build id (the linker's hash of the whole file, read from the
// loaded image: no i/o, so a short program pays nothing for it); without one, a hash of the shared object or executable that holds this
// function, read once. A rebuild is a different simulator and must not replay what the old one wrote.
struct BuildNote {
  const void* addr = nullptr;
  Hash128 id;
  bool found = false;
};

int find_build_id(struct dl_phdr_info* info, size_t, void* data) {
  BuildNote* out = static_cast<BuildNote*>(data);
  const uintptr_t here = reinterpret_cast<uintptr_t>(out->addr);
  bool inside = false;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr)& ph = info->dlpi_phdr[i];
    if (ph.p_type != PT_LOAD) continue;
    const uintptr_t lo = info->dlpi_addr + ph.p_vaddr;
    if (here >= lo && here < lo + ph.p_memsz) inside = true;
  }
  if (!inside) return 0;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr)& ph = info->dlpi_phdr[i];
    if (ph.p_type != PT_NOTE) continue;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(info->dlpi_addr + ph.p_vaddr);
    const uint8_t* end = p + ph.p_memsz;
    while (p + sizeof(ElfW(Nhdr)) <= end) {
      const ElfW(Nhdr)* n = reinterpret_cast<const ElfW(Nhdr)*>(p);
      const uint8_t* name = p + sizeof(ElfW(Nhdr));
      const uint8_t* desc = name + ((n->n_namesz + 3) & ~3u);
      const uint8_t* next = desc + ((n->n_descsz + 3) & ~3u);
      if (next > end) break;
      if (n->n_type == NT_GNU_BUILD_ID && n->n_namesz == 4 && std::memcmp(name, "GNU", 4) == 0 && n->n_descsz > 0) {
        out->id = hash_bytes(desc, n->n_descsz);
        out->found = true;
      }
      p = next;
    }
  }
  return 1;   // this is the object: stop
}

Hash128 build_identity() {
  static const Hash128 id = [] {
    Hasher h(0x6275696c);
#ifdef VGPU_VERSION
    h.str(VGPU_VERSION);
#endif
    BuildNote note;
    note.addr = reinterpret_cast<const void*>(&build_identity);
    dl_iterate_phdr(find_build_id, &note);
    if (note.found) {
      h.hash(note.id);
      return h.done();
    }
    Dl_info info{};
    std::string path = "/proc/self/exe";
    if (dladdr(reinterpret_cast<void*>(&build_identity), &info) && info.dli_fname && *info.dli_fname) path = info.dli_fname;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f && path != "/proc/self/exe") f = std::fopen("/proc/self/exe", "rb");
    if (f) {
      std::vector<uint8_t> buf(1 << 20);
      Hasher body(0x66696c65);
      uint64_t total = 0;
      size_t n;
      while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
        body.hash(hash_bytes(buf.data(), n));
        total += n;
      }
      std::fclose(f);
      body.u64(total);
      h.hash(body.done());
    }
    return h.done();
  }();
  return id;
}

Hash128 fresh_root() {
  static std::atomic<uint64_t> counter{0};   // process-wide: two devices, or two caches in one process, never share a root
  const uint64_t seq = counter.fetch_add(1);
  static const uint64_t seed = [] {
    std::random_device rd;
    return (uint64_t{rd()} << 32) ^ rd() ^ (static_cast<uint64_t>(::getpid()) << 17) ^ now_ns();
  }();
  Hasher h(0x706f6973);
  h.u64(seed), h.u64(seq);
  return h.done();
}

}  // namespace

// What a kernel calls that has an effect outside device memory: the heap, printf, and the device runtime (cudaLaunchDevice,
// cudaGraphLaunch, cudaGetDevice, the cooperative-groups handle, ...). Not ptxas's own helpers for division, remainder and the like,
// which are named __cuda_sm20_div_rn_f64_full and so on and are ordinary code.
bool is_device_runtime_name(const std::string& f) {
  if (f == "malloc" || f == "free" || f == "vprintf" || f == "printf") return true;
  if (f.rfind("cuda", 0) == 0) return true;
  for (const char* p : {"__cudaCDP", "__cudart", "__cuda_syscall"})
    if (f.rfind(p, 0) == 0) return true;
  return false;
}

Config Config::from_env() {
  Config c;
  const char* e = std::getenv("VGPU_MEMO");
  c.enabled = e && e[0] && e[0] != '0';
  if (const char* d = std::getenv("VGPU_MEMO_DIR"); d && *d) c.dir = d;
  c.disk_cap = env_u64("VGPU_MEMO_MAX_MB", 4096) << 20;
  c.mem_cap = env_u64("VGPU_MEMO_MEM_MB", 512) << 20;
  c.entry_cap = env_u64("VGPU_MEMO_ENTRY_MB", 256) << 20;
  if (const char* b = std::getenv("VGPU_MEMO_BUDGET"); b && *b) {
    if (std::strcmp(b, "inf") == 0) c.force_record = true;
    else c.budget = std::atof(b);
  }
  c.verify = static_cast<int>(env_u64("VGPU_MEMO_VERIFY", 0));
  if (const char* e = std::getenv("VGPU_MEMO_EVENT_CREDIT_MS"); e && *e) c.event_credit_ns = static_cast<uint64_t>(std::atof(e) * 1e6);
  if (const char* m = std::getenv("VGPU_MEMO_TEST_MUTANT"); m && *m) c.mutant = m;
  return c;
}

// ---- the store: memory, then disk -------------------------------------------------------------------------------

struct DeviceMemo::Store {
  explicit Store(const Config& c) : cfg(c) {
    if (!cfg.dir.empty()) ::mkdir(cfg.dir.c_str(), 0777);   // (the size of what is there is worked out when something is first written)
  }

  using Ptr = std::shared_ptr<const Entry>;

  struct Item {
    Hash128 key;
    Ptr e;
    uint64_t bytes;
  };
  struct KeyHash {
    size_t operator()(const Hash128& k) const { return static_cast<size_t>(k.a ^ (k.b * 0x9e3779b97f4a7c15ull)); }
  };

  Config cfg;
  // Which keys have a file on disk, learned a bucket (the first two hex digits) at a time, the first time a key of that bucket is
  // asked for: a miss then costs a hash lookup and not a failed open. Files other processes add later are not seen.
  std::unordered_set<Hash128, KeyHash> on_disk;
  bool bucket_loaded[256] = {};
  void load_bucket(unsigned b) {
    if (bucket_loaded[b]) return;
    bucket_loaded[b] = true;
    char sub[16];
    std::snprintf(sub, sizeof sub, "%02x", b);
    if (DIR* d = ::opendir((cfg.dir + "/" + sub).c_str())) {
      while (const dirent* f = ::readdir(d)) {
        if (std::strlen(f->d_name) != 36 || std::strcmp(f->d_name + 32, ".vgm") != 0) continue;
        Hash128 k;
        if (std::sscanf(f->d_name, "%16" SCNx64 "%16" SCNx64, &k.a, &k.b) == 2) on_disk.insert(k);
      }
      ::closedir(d);
    }
  }
  std::list<Item> lru;   // front: most recently used
  std::unordered_map<Hash128, std::list<Item>::iterator, KeyHash> index;
  uint64_t held = 0;
  uint64_t disk_bytes = 0;
  bool disk_scanned = false;
  uint64_t tmp_seq = 0;
  std::mutex io_mu;

  std::string path_for(const Hash128& k) const {
    const std::string h = hex128(k);
    return cfg.dir + "/" + h.substr(0, 2) + "/" + h + ".vgm";
  }

  void scan_disk() {
    uint64_t total = 0;
    if (DIR* d = ::opendir(cfg.dir.c_str())) {
      while (const dirent* ent = ::readdir(d)) {
        if (ent->d_name[0] == '.') continue;
        const std::string sub = cfg.dir + "/" + ent->d_name;
        if (DIR* s = ::opendir(sub.c_str())) {
          while (const dirent* f = ::readdir(s)) {
            if (f->d_name[0] == '.') continue;
            struct stat st;
            if (::stat((sub + "/" + f->d_name).c_str(), &st) == 0) total += static_cast<uint64_t>(st.st_size);
          }
          ::closedir(s);
        }
      }
      ::closedir(d);
    }
    disk_bytes = total;
  }

  Ptr find_memory(const Hash128& k) {
    const auto it = index.find(k);
    if (it == index.end()) return nullptr;
    lru.splice(lru.begin(), lru, it->second);
    return it->second->e;
  }

  void put_memory(const Hash128& k, Ptr e) {
    const uint64_t b = e->bytes();
    if (b > cfg.mem_cap) return;
    if (const auto it = index.find(k); it != index.end()) {
      held -= it->second->bytes;
      lru.erase(it->second);
      index.erase(it);
    }
    lru.push_front(Item{k, std::move(e), b});
    index[k] = lru.begin();
    held += b;
    while (held > cfg.mem_cap && lru.size() > 1) {
      const Item& last = lru.back();
      held -= last.bytes;
      index.erase(last.key);
      lru.pop_back();
    }
  }

  // ---- disk format: header, counters, runs, a hash of all of it ----
  static constexpr char kMagic[8] = {'V', 'G', 'M', 'E', 'M', '0', '3', 0};

  static void put_u64(std::vector<uint8_t>& v, uint64_t x) {
    uint8_t b[8];
    std::memcpy(b, &x, 8);
    v.insert(v.end(), b, b + 8);
  }

  bool write_disk(const Hash128& k, const Entry& e) {
    std::lock_guard<std::mutex> io(io_mu);
    if (!disk_scanned) {
      disk_scanned = true;
      scan_disk();
    }
    std::vector<uint8_t> v;
    v.insert(v.end(), kMagic, kMagic + 8);
    put_u64(v, k.a), put_u64(v, k.b);
    put_u64(v, exec::kCounterWords), put_u64(v, e.stats.inst_by_opcode.size());
    put_u64(v, e.exec_ns), put_u64(v, e.runs.size());
    const uint8_t* cw = reinterpret_cast<const uint8_t*>(&e.stats);
    v.insert(v.end(), cw, cw + exec::kCounterWords * 8);
    for (uint64_t x : e.stats.inst_by_opcode) put_u64(v, x);
    for (const Run& r : e.runs) {
      put_u64(v, r.addr), put_u64(v, r.len), put_u64(v, (r.uniform ? 1u : 0u) | (uint64_t{r.byte} << 8));
      if (!r.uniform) v.insert(v.end(), r.data.begin(), r.data.end());
    }
    const Hash128 sum = hash_bytes(v.data(), v.size());
    put_u64(v, sum.a), put_u64(v, sum.b);
    const std::string final_path = path_for(k);
    ::mkdir((cfg.dir + "/" + hex128(k).substr(0, 2)).c_str(), 0777);
    char tmp[512];
    std::snprintf(tmp, sizeof tmp, "%s.tmp.%d.%" PRIu64, final_path.c_str(), static_cast<int>(::getpid()), tmp_seq++);
    std::FILE* f = std::fopen(tmp, "wb");
    if (!f) return false;
    const bool ok = std::fwrite(v.data(), 1, v.size(), f) == v.size();
    if (std::fclose(f) != 0 || !ok) {
      ::unlink(tmp);
      return false;
    }
    if (::rename(tmp, final_path.c_str()) != 0) {
      ::unlink(tmp);
      return false;
    }
    disk_bytes += v.size();
    on_disk.insert(k);
    return true;
  }

  Ptr read_disk(const Hash128& k) {
    {
      std::lock_guard<std::mutex> io(io_mu);
      load_bucket(static_cast<unsigned>(k.a >> 56));
      if (!on_disk.count(k)) return nullptr;
    }
    std::lock_guard<std::mutex> io(io_mu);
    const std::string p = path_for(k);
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return nullptr;
    std::vector<uint8_t> v;
    {
      std::fseek(f, 0, SEEK_END);
      const long n = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      if (n < 0) {
        std::fclose(f);
        return nullptr;
      }
      v.resize(static_cast<size_t>(n));
      if (std::fread(v.data(), 1, v.size(), f) != v.size()) {
        std::fclose(f);
        return nullptr;
      }
    }
    std::fclose(f);
    const auto bad = [&]() -> Ptr {
      ::unlink(p.c_str());   // a file that does not parse is not an entry
      on_disk.erase(k);
      return nullptr;
    };
    if (v.size() < 8 + 6 * 8 + 16 || std::memcmp(v.data(), kMagic, 8) != 0) return bad();
    const Hash128 sum = hash_bytes(v.data(), v.size() - 16);
    uint64_t sa, sb;
    std::memcpy(&sa, v.data() + v.size() - 16, 8);
    std::memcpy(&sb, v.data() + v.size() - 8, 8);
    if (sum.a != sa || sum.b != sb) return bad();
    size_t at = 8;
    const auto u64 = [&]() {
      uint64_t x;
      std::memcpy(&x, v.data() + at, 8);
      at += 8;
      return x;
    };
    const size_t end = v.size() - 16;
    const uint64_t ka = u64(), kb = u64();
    if (ka != k.a || kb != k.b) return bad();
    const uint64_t words = u64(), nops = u64(), exec_ns = u64(), nruns = u64();
    if (words != exec::kCounterWords) return bad();
    auto e = std::make_shared<Entry>();
    if (at + words * 8 + nops * 8 > end) return bad();
    std::memcpy(static_cast<void*>(&e->stats), v.data() + at, words * 8);
    at += words * 8;
    e->stats.inst_by_opcode.resize(static_cast<size_t>(nops));
    for (uint64_t i = 0; i < nops; ++i) e->stats.inst_by_opcode[i] = u64();
    e->exec_ns = exec_ns;
    for (uint64_t i = 0; i < nruns; ++i) {
      if (at + 24 > end) return bad();
      Run r;
      r.addr = u64(), r.len = u64();
      const uint64_t fl = u64();
      r.uniform = (fl & 1) != 0;
      r.byte = static_cast<uint8_t>(fl >> 8);
      if (!r.uniform) {
        if (r.len > end - at) return bad();
        r.data.assign(v.begin() + static_cast<long>(at), v.begin() + static_cast<long>(at + r.len));
        at += r.len;
      }
      e->runs.push_back(std::move(r));
    }
    if (at != end) return bad();
    // Use is recency: the least recently used entry goes first.
    struct timespec ts[2] = {{0, UTIME_NOW}, {0, UTIME_NOW}};
    ::utimensat(AT_FDCWD, p.c_str(), ts, 0);
    return e;
  }

  void evict_disk(uint64_t* evicted) {
    std::lock_guard<std::mutex> io(io_mu);
    if (cfg.dir.empty() || disk_bytes <= cfg.disk_cap) return;
    struct F {
      std::string path;
      time_t mtime;
      uint64_t size;
    };
    std::vector<F> files;
    uint64_t total = 0;
    if (DIR* d = ::opendir(cfg.dir.c_str())) {
      while (const dirent* ent = ::readdir(d)) {
        if (ent->d_name[0] == '.') continue;
        const std::string sub = cfg.dir + "/" + ent->d_name;
        if (DIR* s = ::opendir(sub.c_str())) {
          while (const dirent* f = ::readdir(s)) {
            if (f->d_name[0] == '.') continue;
            const std::string fp = sub + "/" + f->d_name;
            struct stat st;
            if (::stat(fp.c_str(), &st) == 0) {
              files.push_back({fp, st.st_mtime, static_cast<uint64_t>(st.st_size)});
              total += static_cast<uint64_t>(st.st_size);
            }
          }
          ::closedir(s);
        }
      }
      ::closedir(d);
    }
    std::sort(files.begin(), files.end(), [](const F& a, const F& b) { return a.mtime < b.mtime; });
    const uint64_t goal = cfg.disk_cap / 10 * 9;
    for (const F& f : files) {
      if (total <= goal) break;
      if (::unlink(f.path.c_str()) == 0) {
        total -= f.size;
        ++*evicted;
        const size_t slash = f.path.rfind('/');
        Hash128 k;
        if (slash != std::string::npos && std::sscanf(f.path.c_str() + slash + 1, "%16" SCNx64 "%16" SCNx64, &k.a, &k.b) == 2) on_disk.erase(k);
      }
    }
    disk_bytes = total;
  }
};

// ---- the device's cache -------------------------------------------------------------------------------------------

namespace {

// (leaked, like the lists: a device is destroyed during static destruction, after namespace-scope objects of this file)
std::mutex& reg_mutex() {
  static auto* m = new std::mutex;
  return *m;
}
std::vector<DeviceMemo*>& registry() {
  static auto* r = new std::vector<DeviceMemo*>;
  return *r;
}
void report_all_at_exit();
void register_memo(DeviceMemo* d) {
  static const bool once = [] {
    std::atexit(report_all_at_exit);
    return true;
  }();
  (void)once;
  std::lock_guard<std::mutex> g(reg_mutex());
  registry().push_back(d);
}
void unregister_memo(DeviceMemo* d) {
  std::lock_guard<std::mutex> g(reg_mutex());
  auto& r = registry();
  r.erase(std::remove(r.begin(), r.end(), d), r.end());
}

std::FILE* stats_file(bool* close) {
  *close = false;
  const char* path = std::getenv("VGPU_MEMO_STATS_FILE");
  if (path && *path) {
    if (std::FILE* f = std::fopen(path, "a")) {
      *close = true;
      return f;
    }
  }
  const char* e = std::getenv("VGPU_MEMO_STATS");
  if (e && e[0] && e[0] != '0') return stderr;
  return nullptr;
}

void report_all_at_exit() {
  std::vector<DeviceMemo*> all;
  {
    std::lock_guard<std::mutex> g(reg_mutex());
    all = registry();
  }
  bool close = false;
  std::FILE* f = stats_file(&close);
  if (!f) return;
  for (DeviceMemo* d : all) d->report(f);
  if (close) std::fclose(f);
}

}  // namespace

DeviceMemo::DeviceMemo(MemoryManager& mem, const Config& cfg, const DeviceProfile& profile)
    : mem_(mem), cfg_(cfg), profile_(profile_hash(profile)), build_(build_identity()), mutant_(cfg.mutant) {
  store_ = std::make_unique<Store>(cfg_);
  ptrs_ = false;   // an empty device holds no pointers
  // The empty device is the one root every run of a program starts from: its history is its identity.
  Hasher h(0x726f6f74);
  if (!mut("key_no_profile")) h.hash(profile_);
  h.hash(build_);
  state_ = h.done();
  register_memo(this);
}

DeviceMemo::~DeviceMemo() {
  unregister_memo(this);
  bool close = false;
  if (std::FILE* f = stats_file(&close)) {
    report(f);
    if (close) std::fclose(f);
  }
}

// Every VGPU_ variable but the ones that cannot change what a kernel computes (where output goes, what is printed, the thread
// count, which is in the key as one fact: blocks in order or not). A variable that does change results (VGPU_PTX_CONTRACT, the
// profile, VGPU_DARK_FILL, ...) is in by not being out. Worked out again only when the environment's entries have moved.
Hash128 DeviceMemo::env_hash() {
  uint64_t sig = 0, n = 0;
  for (char** e = environ; e && *e; ++e, ++n) sig = sig * 1000003u + reinterpret_cast<uintptr_t>(*e);
  std::lock_guard<std::mutex> g(mu_);
  if (n == env_n_ && sig == env_sig_ && env_valid_) return env_hash_;
  static const char* const ignore[] = {"VGPU_MEMO",      "VGPU_COUNTERS", "VGPU_KERNEL_DIGEST", "VGPU_QUIET",    "VGPU_THREADS",
                                       "VGPU_TELEMETRY", "VGPU_SESSION",  "VGPU_STATE_DIR",     "VGPU_SASS_LOG", "VGPU_SASS_CACHE",
                                       "VGPU_JIT_STATS", "VGPU_TIER_STATS", "VGPU_HOST_GPU_LOG", "VGPU_REGISTERS_DIR"};
  std::vector<std::string> vars;
  for (char** e = environ; e && *e; ++e) {
    if (std::strncmp(*e, "VGPU_", 5) != 0) continue;
    bool skip = false;
    for (const char* ig : ignore)
      if (std::strncmp(*e, ig, std::strlen(ig)) == 0) skip = true;
    if (!skip) vars.push_back(*e);
  }
  std::sort(vars.begin(), vars.end());
  Hasher h(0x656e7673);
  for (const std::string& v : vars) h.str(v.c_str());
  env_hash_ = h.done();
  env_n_ = n;
  env_sig_ = sig;
  env_valid_ = true;
  return env_hash_;
}

Hash128 DeviceMemo::profile_hash(const DeviceProfile& p) const { return memo::profile_hash(p); }

Hash128 profile_hash(const DeviceProfile& p) {
  Hasher h(0x70726f66);
  h.str(p.id.c_str()), h.str(p.model.c_str()), h.str(p.vendor.c_str()), h.str(p.architecture.c_str());
  h.u64(static_cast<uint64_t>(p.cc_major)), h.u64(static_cast<uint64_t>(p.cc_minor)), h.u64(p.warp_size), h.u64(p.vram_bytes);
  const Limits& l = p.limits;
  h.u64(l.max_threads_per_block);
  for (uint32_t v : l.max_block_dim) h.u64(v);
  for (uint32_t v : l.max_grid_dim) h.u64(v);
  h.u64(l.shared_mem_per_block), h.u64(l.shared_mem_per_block_optin), h.u64(l.registers_per_block), h.u64(l.multiprocessors);
  h.u64(l.registers_per_sm), h.u64(l.max_threads_per_sm), h.u64(l.max_blocks_per_sm), h.u64(l.max_registers_per_thread);
  h.u64(l.shared_mem_per_sm), h.u64(l.l2_cache_bytes);
  for (const auto& [k, v] : p.features) h.str(k.c_str()), h.u64(v ? 1 : 0);
  // The numerics decide result bits (the tensor-core sum, and the model VGPU_TC_EXACT=published would turn on), so two profiles
  // that differ in one of them must not share a cached result. The provenance classes and the evidence text describe the values
  // and change no result: they are not hashed.
  const Numerics& n = p.numerics;
  h.u64(n.present ? 1 : 0), h.u64(static_cast<uint64_t>(n.tensor_sum)), h.u64(n.tensor_sum_chunk), h.u64(n.tensor_sum_chunk_tf32);
  h.u64(static_cast<uint64_t>(static_cast<int64_t>(n.tensor_sum_grid))), h.u64(static_cast<uint64_t>(n.tensor_sum_published));
  h.u64(static_cast<uint64_t>(n.tensor_sum_fp8)), h.u64(n.tensor_sum_fp8_chunk);
  h.u64(static_cast<uint64_t>(static_cast<int64_t>(n.tensor_sum_fp8_grid))), h.u64(static_cast<uint64_t>(static_cast<int64_t>(n.tensor_sum_fp8_out_bits)));
  return h.done();
}

Stats DeviceMemo::stats() const {
  std::lock_guard<std::mutex> g(mu_);
  return stats_;
}

Hash128 DeviceMemo::state() const {
  std::lock_guard<std::mutex> g(mu_);
  return state_;
}

void DeviceMemo::drop_memory_store() {
  std::lock_guard<std::mutex> g(mu_);
  store_->lru.clear();
  store_->index.clear();
  store_->held = 0;
}

void DeviceMemo::note_reason(const char* why) { ++stats_.reasons[why]; }

// ---- events -------------------------------------------------------------------------------------------------------

void DeviceMemo::poison_locked(const char* why) {
  ++alloc_epoch_;
  state_ = fresh_root();
  ptrs_ = true;
  cache_.clear();
  ++stats_.poisoned;
  note_reason(why);
}

// The state after a launch that changed memory: the key (S and the launch), or in the wrong rules the tests must catch,
// a state that forgets the order or does not move.
void DeviceMemo::advance_locked(const Ticket& t) {
  if (mut("state_never_advances")) return;
  if (mut("commutative_chain")) {
    state_.a += t.desc.a, state_.b += t.desc.b;
    return;
  }
  state_ = t.key;
}

void DeviceMemo::poison(const char* why) {
  std::lock_guard<std::mutex> g(mu_);
  poison_locked(why);
}

void DeviceMemo::invalidate_range_locked(uint64_t addr, uint64_t len) {
  if (mut("no_cache_invalidation")) return;
  auto it = cache_.upper_bound(addr);
  if (it != cache_.begin()) --it;
  while (it != cache_.end() && it->first < addr + len) {
    if (it->first + it->second.size > addr) it->second.valid = false;
    ++it;
  }
}

void DeviceMemo::scan_pointers(uint64_t addr, const uint8_t* p, uint64_t len) {
  if (!ptrs_ && has_pointer_like(addr, p, len)) ptrs_ = true;   // (caller holds mu_)
}

// Whether the bytes of a host write of `len` bytes may be hashed. The hashing is part of the budget that recording shares: what the events
// have cost so far, past their credit, counts in the pool the record decision draws on (fixed + record + event), and the events have a cap of their own, the
// same share of the kernel time (VGPU_MEMO_BUDGET) plus their own credit, so that a program that uploads more than its credit between
// short launches cannot spend the pool on hashing. A write that does not fit is not hashed, and the caller poisons the history instead:
// the history then names a state no other run has, so it can never produce a hit that is not the right one, and what is lost is the
// hits after it.
bool DeviceMemo::event_budget_allows(uint64_t len) {
  if (cfg_.force_record) return true;
  std::lock_guard<std::mutex> g(mu_);
  const double est = static_cast<double>(len) / bw_ * 1e9 + 300.0;
  const double allowed = cfg_.budget * static_cast<double>(stats_.exec_ns) + static_cast<double>(cfg_.event_credit_ns);
  return static_cast<double>(stats_.event_ns) + est <= allowed;
}

void DeviceMemo::on_write(uint64_t dst, const void* src, uint64_t len) {
  if (t_scope == kReplay) return;
  if (t_scope == kEngine) return;   // a kernel's store: the launch's diff has it
  if (dormant_.load(std::memory_order_relaxed) && !mut("dormant_keeps_history")) return;   // the cache is off for good: no history is kept
  if (mut("no_write_event")) return;
  constexpr uint64_t kTimed = 1u << 16;   // (smaller writes are charged by estimate: a clock read costs a tenth of one)
  const uint64_t t0 = len >= kTimed ? now_ns() : 0;
  if (src && !event_budget_allows(len)) {
    if (!mut("event_budget_no_poison")) {
      std::lock_guard<std::mutex> g(mu_);
      ++event_seq_;
      ++stats_.events_skipped;
      poison_locked("event-budget");
      return;
    }
    // (the wrong rule, for the tests: the write is an event without its bytes)
    std::lock_guard<std::mutex> g(mu_);
    Hasher h(0x77726974);
    h.hash(state_), h.u64(dst), h.u64(len), h.hash(Hash128{1, 2});
    state_ = h.done();
    ++event_seq_;
    invalidate_range_locked(dst, len);
    return;
  }
  const Hash128 bytes = src ? hash_bytes(src, static_cast<size_t>(len)) : Hash128{1, 2};
  const uint64_t t1 = len >= kTimed ? now_ns() : 0;
  std::lock_guard<std::mutex> g(mu_);
  Hasher h(0x77726974);
  h.hash(state_), h.u64(dst), h.u64(len), h.hash(bytes);
  state_ = h.done();
  ++event_seq_;
  invalidate_range_locked(dst, len);
  if (src) scan_pointers(dst, static_cast<const uint8_t*>(src), len);
  if (len >= kTimed) {
    const uint64_t t2 = now_ns();
    stats_.event_ns += t2 - t0;
    if (len >= (1u << 20) && t1 > t0) bw_ = 0.7 * bw_ + 0.3 * (static_cast<double>(len) / static_cast<double>(t1 - t0) * 1e9);
  } else {
    stats_.event_ns += 300 + static_cast<uint64_t>(static_cast<double>(len) / bw_ * 1e9);
  }
}

void DeviceMemo::on_fill(uint64_t dst, const uint8_t* pattern, uint32_t pattern_len, uint64_t len) {
  if (t_scope != kHost) return;
  if (mut("no_fill_event")) return;
  std::lock_guard<std::mutex> g(mu_);
  Hasher h(0x66696c6c);
  h.hash(state_), h.u64(dst), h.u64(len), h.bytes(pattern, pattern_len);
  state_ = h.done();
  ++event_seq_;
  invalidate_range_locked(dst, len);
  if (!ptrs_ && pattern_len && pattern_len <= 8 && 8 % pattern_len == 0) {
    uint8_t w[8];
    for (int i = 0; i < 8; ++i) w[i] = pattern[static_cast<uint32_t>(i) % pattern_len];
    uint64_t v;
    std::memcpy(&v, w, 8);
    if (is_device_ptr(v)) ptrs_ = true;
  }
}

void DeviceMemo::on_copy(uint64_t dst, uint64_t src, uint64_t len) {
  if (t_scope != kHost) return;
  if (mut("no_copy_event")) return;
  std::lock_guard<std::mutex> g(mu_);
  Hasher h(0x636f7079);
  h.hash(state_), h.u64(dst), h.u64(src), h.u64(len);
  state_ = h.done();
  ++event_seq_;
  invalidate_range_locked(dst, len);
}

void DeviceMemo::on_alloc(uint64_t base, uint64_t size) {
  if (t_scope == kReplay) return;
  if (t_scope == kEngine) {
    if (!mut("no_poison_taint")) tainted_.store(true, std::memory_order_relaxed);
    return;
  }
  if (mut("no_alloc_event")) return;
  std::lock_guard<std::mutex> g(mu_);
  Hasher h(0x616c6c6f);
  h.hash(state_), h.u64(base), h.u64(size);
  state_ = h.done();
  ++event_seq_;
  ++alloc_epoch_;
}

void DeviceMemo::on_free(uint64_t ptr) {
  if (t_scope == kReplay) return;
  if (t_scope == kEngine) {
    if (!mut("no_poison_taint")) tainted_.store(true, std::memory_order_relaxed);
    return;
  }
  if (mut("no_free_event")) return;
  std::lock_guard<std::mutex> g(mu_);
  Hasher h(0x66726565);
  h.hash(state_), h.u64(ptr);
  state_ = h.done();
  ++event_seq_;
  ++alloc_epoch_;
  cache_.erase(ptr);
  mat_hint_.erase(ptr);
}

void DeviceMemo::on_poison(const char* why) {
  if (t_scope == kEngine) {
    if (!mut("no_poison_taint")) tainted_.store(true, std::memory_order_relaxed);
    return;
  }
  if (t_scope == kReplay) return;
  if (mut("no_poison_vmm")) return;
  std::lock_guard<std::mutex> g(mu_);
  ++event_seq_;
  poison_locked(why);
}

void DeviceMemo::forget_kernels() {
  std::lock_guard<std::mutex> g(mu_);
  device_calls_.clear();
}

// ---- which launches can be cached ----------------------------------------------------------------------------------

bool DeviceMemo::kernel_has_device_calls(const ptx::EntryFn& fn) {
  {
    std::lock_guard<std::mutex> g(mu_);
    const auto it = device_calls_.find(&fn);
    if (it != device_calls_.end()) return it->second.calls;
  }
  // A kernel that mallocs or prints, or calls the device runtime (cuda*: launches, graph launches, conditionals,
  // attribute queries), has effects outside memory that a replay would skip.
  const auto bad = [](const std::string& f) { return is_device_runtime_name(f); };
  bool found = false;
  if (fn.sass)
    for (const std::string& f : sass::reachable(*fn.sass, fn.name))
      if (bad(f)) found = true;
  std::set<const ptx::EntryFn*> seen;
  std::vector<const ptx::EntryFn*> todo{&fn};
  while (!todo.empty() && !found) {
    const ptx::EntryFn* f = todo.back();
    todo.pop_back();
    if (!seen.insert(f).second) continue;
    for (const auto& ins : f->body)
      if (const auto* c = std::get_if<ptx::OpCall>(&ins.op)) {
        if (bad(c->callee)) found = true;
        if (c->target) todo.push_back(c->target);
      }
  }
  std::lock_guard<std::mutex> g(mu_);
  KernelInfo& ki = device_calls_[&fn];
  ki.calls = found;
  Hasher nh(0x6e616d65);
  nh.str(fn.name.c_str());
  ki.name_hash = nh.done();
  return found;
}

const char* DeviceMemo::ineligible(const KernelRef& k, Ticket& t) {
  const exec::LaunchConfig& c = *k.cfg;
  if (k.device_count > 1) return "multi-device";
  if (!k.symbols_known) return "unknown-symbols";
  if (!k.code_ok) return "unhashable-module";
  if (c.cooperative || c.coop_workspace) return "cooperative";
  if (c.cluster != std::array<uint32_t, 3>{0, 0, 0} || (k.fn && k.fn->req_cluster != std::array<uint32_t, 3>{0, 0, 0})) return "cluster";
  if (c.warp_size) return "warp-size";
  if (c.textures && !c.textures->empty() && !mut("no_poison_textures")) return "textures";
  if (c.current_graph_exec) return "device-graph";   // (a graph instantiated for device launch: its kernels may read their own handle)
  const exec::LaunchEnv& e = *k.env;
  if (e.race) return "env-race";
  if (e.trace || e.sass_trace) return "env-trace";
  if (e.max_steps) return "env-max-steps";
  if (e.fail_alloc || e.fail_launch || e.fail_memcpy) return "env-fail-injection";
  if (e.tier_stats) return "env-tier-stats";
  if (profiling::enabled() || profiling::hooked()) return "profiling";
  if (MemoryManager::memo_trace_active()) return "mem-trace";
  if (mem_.memo_faults_armed() && !mut("no_poison_faults")) return "faults-armed";
  if (mem_.memo_untrackable() && !mut("no_poison_vmm")) return "vmm-or-shared-memory";
  if (k.fn && kernel_has_device_calls(*k.fn) && !mut("no_poison_device_calls")) return "device-runtime-call";
  if (active_launches_.load(std::memory_order_relaxed) != 0 && !mut("no_poison_concurrent")) return "concurrent-launch";
  const char* thr = k.env->threads;
  t.sequential = thr && std::strcmp(thr, "1") == 0;
  return nullptr;
}

// ---- the key ---------------------------------------------------------------------------------------------------------

Hash128 DeviceMemo::launch_key(const KernelRef& k, const Hash128& state, const Ticket& t) {
  const exec::LaunchConfig& c = *k.cfg;
  Hasher h(0x6c61756e);
  if (!mut("key_no_state")) h.hash(state);
  if (!mut("key_no_code")) h.hash(k.code);
  {
    // (the name's hash is worked out once per kernel, by kernel_has_device_calls)
    std::lock_guard<std::mutex> g(mu_);
    const auto it = device_calls_.find(k.fn);
    if (k.fn && it != device_calls_.end()) h.hash(it->second.name_hash);
    else h.str(k.name->c_str());
  }
  h.u64(k.sass ? 1 : 0);
  if (!mut("key_no_profile")) h.hash(profile_);
  h.hash(build_);
  if (!mut("key_no_grid")) {
    for (uint32_t v : c.grid) h.u64(v);
    for (uint32_t v : c.block) h.u64(v);
  }
  if (!mut("key_no_shared")) h.u64(c.shared_bytes);
  h.u64(c.work_dim), h.u64(static_cast<uint64_t>(c.scheduler)), h.u64(c.scheduler_seed), h.u64(c.max_steps);
  for (uint32_t v : c.global_offset) h.u64(v);
  h.u64(c.device_heap_bytes), h.u64(c.stack_bytes), h.u64(c.nonportable_cluster ? 1 : 0), h.u64(c.graph_launcher ? 1 : 0);
  if (!mut("key_no_args")) {
    h.u64(k.args->size());
    for (const auto& a : *k.args) h.bytes(a.data(), a.size());
  }
  if (!mut("key_no_env")) {
    // Every variable an engine reads per launch, and whether blocks run in order (the one thread count that matters).
    const exec::LaunchEnv& e = *k.env;
    const char* vars[] = {e.fastpath, e.tc_exact, e.strict, e.lockstep, e.scheduler, e.scheduler_seed, e.tier_chain};
    for (const char* v : vars) {
      h.u64(v ? 1 : 0);
      if (v) h.str(v);
    }
    h.u64(t.sequential ? 1 : 0);
    h.hash(env_hash());
  }
  return h.done();
}

// ---- what a launch could write -----------------------------------------------------------------------------------------

// The allocations a kernel can reach: every allocation an argument word points into, and the module's globals; when
// device pointers may be stored in memory (a pointer table, a linked structure), every live allocation. (Pinned and managed
// memory is not in the list: a kernel that touches any is found out afterwards, by the memory manager, and poisons.)
bool DeviceMemo::collect_bases(const KernelRef& k, Ticket& t) {
  std::set<uint64_t> bases;
  bool all;
  {
    std::lock_guard<std::mutex> g(mu_);
    all = ptrs_ && !mut("ptrs_never");
  }
  const auto note = [&](uint64_t v) {
    if (is_device_ptr(v)) {
      uint64_t base = 0, size = 0;
      if (mem_.find_allocation(v, &base, &size)) bases.insert(base);
    }
  };
  for (const auto& a : *k.args) {
    const size_t step = a.size() <= 512 ? 1 : 4;
    for (size_t i = 0; i + 8 <= a.size(); i += step) {
      uint64_t w;
      std::memcpy(&w, a.data() + i, 8);
      note(w);
    }
  }
  if (k.args->size() > 1) {
    // A pointer a kernel assembles from two 32-bit parameters sits across two arguments: the arguments end to end, a word at a time.
    static thread_local std::vector<uint8_t> all;
    all.clear();
    for (const auto& a : *k.args) all.insert(all.end(), a.begin(), a.end());
    for (size_t i = 0; i + 8 <= all.size() && i < 4096; i += 4) {
      uint64_t w;
      std::memcpy(&w, all.data() + i, 8);
      note(w);
    }
  }
  if (k.module_globals && !mut("skip_module_globals"))
    for (uint64_t v : *k.module_globals) note(v);
  if (all) {
    for (const auto& [base, size] : mem_.live_ranges()) bases.insert(base), (void)size;
  }
  t.bases.assign(bases.begin(), bases.end());
  return true;
}

uint64_t DeviceMemo::estimate_ns(const Ticket& t) {
  uint64_t bytes = 0;
  std::vector<uint64_t> unknown;
  {
    std::lock_guard<std::mutex> g(mu_);
    for (uint64_t base : t.bases) {
      const auto h = mat_hint_.find(base);
      if (h == mat_hint_.end()) {
        unknown.push_back(base);
        continue;
      }
      bytes += h->second;                                              // after
      const auto it = cache_.find(base);
      if (it == cache_.end() || !it->second.valid) bytes += h->second;  // and before
    }
  }
  for (uint64_t base : unknown) {   // (a walk of the chunk table, once per allocation)
    // Memory nobody has written yet reads as zero and costs nothing to hash, but a launch is about to write it: the hashed bytes
    // after it are what it writes, so an allocation not yet seen is taken to be at least half full.
    uint64_t bb = 0, sz = 0;
    mem_.find_allocation(base, &bb, &sz);
    const uint64_t m = std::max(mem_.memo_materialized(base), sz / 2);
    std::lock_guard<std::mutex> g(mu_);
    mat_hint_[base] = m;
    bytes += 2 * m;
  }
  return static_cast<uint64_t>(static_cast<double>(bytes) / bw_ * 1e9) + 3000 + 400 * t.bases.size();
}

// What recording would cost when a kernel may reach every allocation: from the sizes seen at the last walk of all of them, which
// is renewed when allocations change and every so often otherwise (an estimate, never a claim).
uint64_t DeviceMemo::estimate_all_ns() {
  {
    std::lock_guard<std::mutex> g(mu_);
    if (all_est_epoch_ == alloc_epoch_ && all_est_age_ < 64) {
      ++all_est_age_;
      return static_cast<uint64_t>(static_cast<double>(2 * all_est_bytes_) / bw_ * 1e9) + 3000;
    }
  }
  uint64_t bytes = 0;
  size_t n = 0;
  for (const auto& [base, size] : mem_.live_ranges()) {
    const uint64_t m = std::max(mem_.memo_materialized(base), size / 2);
    bytes += m;
    ++n;
    std::lock_guard<std::mutex> g(mu_);
    mat_hint_[base] = m;
  }
  std::lock_guard<std::mutex> g(mu_);
  all_est_bytes_ = bytes;
  all_est_epoch_ = alloc_epoch_;
  all_est_age_ = 0;
  return static_cast<uint64_t>(static_cast<double>(2 * bytes) / bw_ * 1e9) + 3000 + 400 * n;
}

// The before-state of every allocation in t.bases, kept from the last launch's after-state when nothing touched it since.
bool DeviceMemo::capture_before(Ticket& t, uint64_t* hashed) {
  for (uint64_t base : t.bases) {
    {
      std::lock_guard<std::mutex> g(mu_);
      const auto it = cache_.find(base);
      if (it != cache_.end() && it->second.valid) continue;
    }
    std::vector<uint64_t> pages;
    uint64_t mat = 0;
    if (!mem_.memo_page_hashes(base, &pages, &mat)) return false;
    *hashed += mat;
    uint64_t sz = 0, bb = 0;
    mem_.find_allocation(base, &bb, &sz);   // (outside mu_: the table's lock is taken inside alloc, which calls on_alloc)
    std::lock_guard<std::mutex> g(mu_);
    mat_hint_[base] = mat;
    PageCache& pc = cache_[base];
    pc.pages = std::move(pages);
    pc.size = sz;
    pc.valid = true;
  }
  return true;
}

// ---- the launch ----------------------------------------------------------------------------------------------------------

bool DeviceMemo::begin(const KernelRef& k, Ticket& t, exec::LaunchStats* out_stats) {
  const uint64_t t0 = now_ns();
  const bool hit = begin_impl(k, t, out_stats);
  if (t.active) t.begin_ns = now_ns() - t0;
  return hit;
}

bool DeviceMemo::begin_impl(const KernelRef& k, Ticket& t, exec::LaunchStats* out_stats) {
  t = Ticket{};
  if (dormant_.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> g(mu_);
    ++stats_.dormant_launches;
    return false;
  }
  const uint64_t t0 = now_ns();
  t.t_begin = t0;
  t.active = true;
  {
    std::lock_guard<std::mutex> g(mu_);
    ++stats_.launches;
  }
  const char* why = ineligible(k, t);
  if (!why) {
    const exec::LaunchConfig& c = *k.cfg;
    t.blocks = uint64_t{c.grid[0]} * c.grid[1] * c.grid[2];
  }
  if (why) {
    t.why = why;
    active_launches_.fetch_add(1);
    tainted_.store(false, std::memory_order_relaxed);
    return false;
  }
  Hash128 state;
  {
    std::lock_guard<std::mutex> g(mu_);
    state = state_;
    t.seq0 = event_seq_;
  }
  t.key = launch_key(k, state, t);
  t.desc = mut("commutative_chain") ? launch_key(k, Hash128{}, t) : t.key;
  t.eligible = true;
  t.why = *k.name;

  // A stored result?
  std::shared_ptr<const Entry> e;
  bool from_disk = false;
  {
    std::lock_guard<std::mutex> g(mu_);
    e = store_->find_memory(t.key);
  }
  if (!e && !cfg_.dir.empty()) {
    std::shared_ptr<const Entry> d = store_->read_disk(t.key);   // (file i/o outside the lock; the store is touched by one launching thread)
    if (d) {
      e = d;
      from_disk = true;
      std::lock_guard<std::mutex> g(mu_);
      store_->put_memory(t.key, d);
    }
  }
  if (e && cfg_.verify == 0) {
    if (replay(*e)) {
      const uint64_t dt = now_ns() - t0;
      std::lock_guard<std::mutex> g(mu_);
      ++stats_.hits;
      if (from_disk) ++stats_.disk_hits;
      stats_.hit_ns += dt;
      stats_.saved_ns += e->exec_ns > dt ? e->exec_ns - dt : 0;
      stats_.replay_bytes += e->bytes();
      if (!e->runs.empty()) advance_locked(t);
      *out_stats = e->stats;
      if (mut("hit_zero_counters")) *out_stats = exec::LaunchStats{};
      stats_.replayed_instructions += out_stats->instructions;
      t.active = false;
      return true;
    }
    std::lock_guard<std::mutex> g(mu_);
    ++stats_.replay_refused;
    e.reset();
  }
  if (e) t.verify_against = e;   // verify: run it for real and compare

  // A miss. Is capturing its write set worth what it costs?
  bool record = true;
  bool all;
  {
    std::lock_guard<std::mutex> g(mu_);
    all = ptrs_ && !mut("ptrs_never");
  }
  bool have_bases = false;
  if (!cfg_.force_record && !t.verify_against) {
    uint64_t est;
    if (all) {
      est = estimate_all_ns();   // (without listing every allocation: a launch not recorded does not need the list)
    } else {
      collect_bases(k, t);
      have_bases = true;
      est = estimate_ns(t);
    }
    std::lock_guard<std::mutex> g(mu_);
    const double allowed = cfg_.budget * static_cast<double>(stats_.exec_ns) + static_cast<double>(cfg_.credit_ns);
    // (the hashing of host writes draws on the same pool, past the credit of its own: uploads the program does before its first launch are free)
    const uint64_t event_over = stats_.event_ns > cfg_.event_credit_ns ? stats_.event_ns - cfg_.event_credit_ns : 0;
    record = static_cast<double>(stats_.fixed_ns + stats_.record_ns + event_over + est) <= allowed;
  }
  if (record && !have_bases) collect_bases(k, t);
  t.record = record;
  if (record) {
    uint64_t hashed = 0;
    const uint64_t p0 = now_ns();
    if (!capture_before(t, &hashed)) {
      t.record = false;
    } else {
      const uint64_t dp = now_ns() - p0;
      t.t_pre = dp;
      if (hashed > (1u << 20) && dp) {
        std::lock_guard<std::mutex> g(mu_);
        bw_ = 0.7 * bw_ + 0.3 * (static_cast<double>(hashed) / static_cast<double>(dp) * 1e9);
      }
    }
  }
  active_launches_.fetch_add(1);
  tainted_.store(false, std::memory_order_relaxed);
  g_smid_by_worker.store(false, std::memory_order_relaxed);
  g_reads_gridid.store(false, std::memory_order_relaxed);
  mem_.memo_host_touched(true);   // (cleared: end() asks whether this launch reached pinned or managed memory)
  return false;
}

bool DeviceMemo::replay(const Entry& e) {
  // Every write is checked before the first is made: a refused replay leaves memory as it was.
  for (const Run& r : e.runs) {
    uint64_t base = 0, size = 0;
    if (!mem_.find_allocation(r.addr, &base, &size) || r.addr + r.len > base + size || r.addr + r.len < r.addr) return false;
    if (mem_.memo_host_maps() && (mem_.memo_is_host_addr(r.addr) || mem_.memo_is_host_addr(r.addr + r.len - 1))) return false;
  }
  try {
    ScopeGuard scope(kReplay);
    for (const Run& r : e.runs) {
      if (r.uniform) mem_.fill(r.addr, &r.byte, 1, r.len);
      else mem_.write(r.addr, r.data.data(), r.len);
    }
  } catch (...) {
    poison("replay-failed");   // memory may be half written
    throw;
  }
  std::lock_guard<std::mutex> g(mu_);
  for (const Run& r : e.runs) {
    invalidate_range_locked(r.addr, r.len);
    if (!r.uniform) scan_pointers(r.addr, r.data.data(), r.len);   // a replayed pointer table is a pointer table
  }
  return true;
}

void DeviceMemo::abort(Ticket& t, const char* why) {
  if (!t.active) return;
  t.active = false;
  active_launches_.fetch_sub(1);
  poison(why);
}

void DeviceMemo::end(Ticket& t, const exec::LaunchStats& stats, uint64_t exec_ns) {
  if (!t.active) return;
  t.active = false;
  active_launches_.fetch_sub(1);
  const uint64_t t_end0 = now_ns();
  // An ineligible launch ran; what it did to memory is not described by anything.
  if (!t.eligible) {
    std::lock_guard<std::mutex> g(mu_);
    stats_.exec_ns += exec_ns;
    stats_.fixed_ns += t.begin_ns + (now_ns() - t_end0);
    ++executed_;
    poison_locked(t.why.c_str());
    return;
  }
  bool concurrent = false, taint = tainted_.load(std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> g(mu_);
    concurrent = event_seq_ != t.seq0;
  }
  const char* poison_why = nullptr;
  if (concurrent && !mut("no_poison_concurrent")) poison_why = "concurrent-host-event";
  else if (taint) poison_why = "device-side-alloc-or-free";
  else if (mem_.memo_host_touched(false) && !mut("no_poison_host_maps")) poison_why = "touched-host-memory";
  else if (g_reads_gridid.load(std::memory_order_relaxed)) poison_why = "reads-gridid";
  else if (stats.atomics > 0 && !t.sequential && !mut("no_poison_atomics")) poison_why = "atomics-on-several-threads";
  else if (g_smid_by_worker.load(std::memory_order_relaxed) && !t.sequential && t.blocks > 1 && !mut("no_poison_smid"))
    poison_why = "smid-by-host-thread";
  if (poison_why) {
    std::lock_guard<std::mutex> g(mu_);
    stats_.exec_ns += exec_ns;
    stats_.fixed_ns += t.begin_ns;
    ++executed_;
    poison_locked(poison_why);
    return;
  }

  if (!t.record) {
    // Ran, not captured: the kernel is deterministic as far as the cache knows, so the state advances by the launch and
    // the before-states kept so far are stale.
    std::lock_guard<std::mutex> g(mu_);
    stats_.exec_ns += exec_ns;
    stats_.fixed_ns += t.begin_ns + (now_ns() - t_end0);
    ++stats_.unrecorded;
    ++executed_;
    advance_locked(t);
    cache_.clear();
    ptrs_ = true;
    return;
  }

  // The after-state, and what changed.
  Entry entry;
  entry.stats = stats;
  entry.exec_ns = exec_ns;
  uint64_t hashed = 0;
  bool ok = true;
  struct Page {
    uint64_t base;
    uint64_t addr;
    uint64_t len;
  };
  std::vector<Page> changed;
  std::map<uint64_t, std::vector<uint64_t>> after;
  for (uint64_t base : t.bases) {
    std::vector<uint64_t> pages;
    uint64_t mat = 0;
    if (!mem_.memo_page_hashes(base, &pages, &mat)) {
      ok = false;
      break;
    }
    hashed += mat;
    std::lock_guard<std::mutex> g(mu_);
    mat_hint_[base] = mat;
    const auto it = cache_.find(base);
    if (it == cache_.end() || !it->second.valid || it->second.pages.size() != pages.size()) {
      ok = false;
      break;
    }
    uint64_t sz = it->second.size;
    for (size_t i = 0; i < pages.size(); ++i)
      if (pages[i] != it->second.pages[i]) {
        const uint64_t off = static_cast<uint64_t>(i) * MemoryManager::kMemoPage;
        changed.push_back({base, base + off, std::min<uint64_t>(MemoryManager::kMemoPage, sz - off)});
      }
    after[base] = std::move(pages);
  }
  if (!ok) {
    // The allocation set moved under the launch (a free the launch could not have made): no entry, no claim.
    std::lock_guard<std::mutex> g(mu_);
    stats_.exec_ns += exec_ns;
    ++executed_;
    poison_locked("allocation-set-changed-in-launch");
    return;
  }
  // Runs: consecutive changed pages read as one.
  uint64_t total = 0;
  bool new_ptrs_unknown = false;
  std::vector<uint8_t> buf;
  for (size_t i = 0; i < changed.size();) {
    size_t j = i + 1;
    uint64_t len = changed[i].len;
    while (j < changed.size() && changed[j].base == changed[i].base && changed[j].addr == changed[j - 1].addr + changed[j - 1].len &&
           changed[j - 1].len == MemoryManager::kMemoPage) {
      len += changed[j].len;
      ++j;
    }
    if (total + len + 32 > cfg_.entry_cap) {
      total = cfg_.entry_cap + 1;   // too big to keep: not read
      break;
    }
    Run r;
    r.addr = changed[i].addr;
    r.len = len;
    buf.resize(static_cast<size_t>(len));
    mem_.read(r.addr, buf.data(), len);
    bool uni = true;
    for (uint64_t x = 1; x < len && uni; ++x) uni = buf[x] == buf[0];
    if (uni) {
      r.uniform = true;
      r.byte = buf[0];
    } else {
      r.data = buf;
    }
    total += uni ? 32 : len + 32;
    if (total > cfg_.entry_cap) break;
    entry.runs.push_back(std::move(r));
    i = j;
  }
  const bool oversize = total > cfg_.entry_cap;
  if (oversize) new_ptrs_unknown = true;
  if (mut("drop_byte") && !entry.runs.empty()) {
    Run& r = entry.runs.back();
    if (!r.uniform && !r.data.empty()) r.data.back() ^= 0x5a;
    else if (r.uniform) r.byte ^= 0x5a;
  }

  // What the pointer scan of the new bytes says about device pointers living in memory.
  bool new_ptrs = new_ptrs_unknown;   // (a write set cut short at the cap was not scanned whole)
  for (const Run& r : entry.runs)
    if (!r.uniform && has_pointer_like(r.addr, r.data.data(), r.len)) new_ptrs = true;

  const bool unchanged = changed.empty();
  uint64_t stored = 0;
  bool verify_bad = false;
  if (t.verify_against) {
    const Entry& s = *t.verify_against;
    bool same = s.runs.size() == entry.runs.size() && std::memcmp(&s.stats, &entry.stats, exec::kCounterWords * 8) == 0 &&
                s.stats.inst_by_opcode == entry.stats.inst_by_opcode;
    for (size_t i = 0; same && i < s.runs.size(); ++i) {
      const Run &a = s.runs[i], &b = entry.runs[i];
      same = a.addr == b.addr && a.len == b.len && a.uniform == b.uniform && (a.uniform ? a.byte == b.byte : a.data == b.data);
    }
    verify_bad = !same;
  }
  if (!oversize && !t.verify_against) {
    auto sp = std::make_shared<Entry>(std::move(entry));
    {
      std::lock_guard<std::mutex> g(mu_);
      store_->put_memory(t.key, sp);
    }
    if (!cfg_.dir.empty()) {
      if (store_->write_disk(t.key, *sp)) stored = sp->bytes();
      uint64_t ev = 0;
      store_->evict_disk(&ev);
      std::lock_guard<std::mutex> g(mu_);
      stats_.disk_evicted += ev;
    }
  }
  const uint64_t t_done = now_ns();
  {
    std::lock_guard<std::mutex> g(mu_);
    stats_.exec_ns += exec_ns;
    ++executed_;
    const uint64_t rec = t.t_pre + (t_done - t_end0);
    stats_.record_ns += rec;
    stats_.fixed_ns += t.begin_ns > t.t_pre ? t.begin_ns - t.t_pre : 0;
    ++stats_.recorded;
    if (oversize) ++stats_.entries_oversize;
    else if (!t.verify_against) ++stats_.entries_put, stats_.entry_bytes += stored;
    if (t.verify_against) {
      ++stats_.verify_checked;
      if (verify_bad) ++stats_.verify_mismatch;
    }
    if (hashed > (1u << 20) && t_done > t_end0) bw_ = 0.7 * bw_ + 0.3 * (static_cast<double>(hashed) / static_cast<double>(t_done - t_end0) * 1e9);
    if (new_ptrs) ptrs_ = true;
    // A launch that changed no byte leaves the state as it was; one that did moves it by the launch.
    if (!unchanged) advance_locked(t);
    stats_.recorded_instructions += stats.instructions;
    for (auto& [base, pages] : after) {
      PageCache& pc = cache_[base];
      pc.pages = std::move(pages);
      pc.valid = true;
    }
    // Adaptive tiers: a device whose launches are too small for the fixed cost to pay goes quiet (the state is poisoned so
    // nothing later can match across the gap).
    if (executed_ >= cfg_.dormant_after && stats_.hits == 0 && static_cast<double>(stats_.fixed_ns) > cfg_.dormant_fixed_share * static_cast<double>(stats_.exec_ns) &&
        !cfg_.force_record && !cfg_.verify) {
      dormant_.store(true, std::memory_order_relaxed);
      poison_locked("dormant-fixed-cost");
    }
  }
  if (verify_bad) {
    std::fprintf(stderr, "[vgpu][memo] VERIFY MISMATCH: kernel %s, key %s: the stored result differs from a real run\n", t.why.c_str(),
                 hex128(t.key).c_str());
    if (cfg_.verify == 1) std::abort();
  }
}

// ---- reporting -----------------------------------------------------------------------------------------------------------

Hash128 memory_hash(const MemoryManager& mem) {
  Hasher h(0x6d656d68);
  for (const auto& [base, size] : mem.live_ranges()) {
    std::vector<uint64_t> pages;
    uint64_t mat = 0;
    h.u64(base), h.u64(size);
    if (mem.memo_page_hashes(base, &pages, &mat))
      for (uint64_t p : pages) h.u64(p);
  }
  return h.done();
}

Hash128 DeviceMemo::memory_hash() const { return vgpu::memo::memory_hash(mem_); }

// VGPU_MEMO_FINAL_HASH=<file>: at the end of the process (or of the device) one line per device with the hash of all its
// live memory, whether or not the cache is on: what the differential runs compare.
namespace {
std::mutex& fh_mutex() {
  static auto* m = new std::mutex;
  return *m;
}
std::vector<const MemoryManager*>& fh_list() {
  static auto* v = new std::vector<const MemoryManager*>;
  return *v;
}
void fh_write(const MemoryManager* m) {
  const char* p = std::getenv("VGPU_MEMO_FINAL_HASH");
  if (!p || !*p) return;
  if (std::FILE* o = std::fopen(p, "a")) {
    std::fprintf(o, "%016llx %s\n", static_cast<unsigned long long>(m->va_base()), hex128(memory_hash(*m)).c_str());
    std::fclose(o);
  }
}
void fh_atexit() {
  std::lock_guard<std::mutex> g(fh_mutex());
  for (const MemoryManager* m : fh_list()) fh_write(m);
  fh_list().clear();
}
}  // namespace

void final_hash_register(const MemoryManager* m) {
  static const bool once = [] {
    std::atexit(fh_atexit);
    return true;
  }();
  (void)once;
  std::lock_guard<std::mutex> g(fh_mutex());
  fh_list().push_back(m);
}

void final_hash_write(const MemoryManager* m) {
  if (!std::getenv("VGPU_MEMO_FINAL_HASH")) return;
  std::lock_guard<std::mutex> g(fh_mutex());
  auto& v = fh_list();
  const auto it = std::find(v.begin(), v.end(), m);
  if (it == v.end()) return;
  fh_write(m);
  v.erase(it);
}

void DeviceMemo::report(std::FILE* f) const {
  if (reported_.exchange(true)) return;   // once: the destructor and the exit handler may both come
  const Stats s = stats();
  std::string reasons;
  for (const auto& [k, v] : s.reasons) reasons += (reasons.empty() ? "" : ",") + k + ":" + std::to_string(v);
  std::fprintf(f,
               "[vgpu][memo] launches=%" PRIu64 " hits=%" PRIu64 " disk_hits=%" PRIu64 " recorded=%" PRIu64 " unrecorded=%" PRIu64 " poisoned=%" PRIu64
               " dormant_launches=%" PRIu64 " replay_refused=%" PRIu64 " verify_checked=%" PRIu64 " verify_mismatch=%" PRIu64 " entries_put=%" PRIu64
               " entry_bytes=%" PRIu64 " exec_ms=%.3f fixed_ms=%.3f record_ms=%.3f event_ms=%.3f events_skipped=%" PRIu64 " hit_ms=%.3f saved_ms=%.3f reasons=%s\n",
               s.launches, s.hits, s.disk_hits, s.recorded, s.unrecorded, s.poisoned, s.dormant_launches, s.replay_refused, s.verify_checked,
               s.verify_mismatch, s.entries_put, s.entry_bytes, static_cast<double>(s.exec_ns) / 1e6, static_cast<double>(s.fixed_ns) / 1e6,
               static_cast<double>(s.record_ns) / 1e6, static_cast<double>(s.event_ns) / 1e6, s.events_skipped, static_cast<double>(s.hit_ns) / 1e6, static_cast<double>(s.saved_ns) / 1e6,
               reasons.empty() ? "-" : reasons.c_str());
}

}  // namespace vgpu::memo
