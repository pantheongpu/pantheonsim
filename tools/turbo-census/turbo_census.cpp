// turbo_census: the refusal census of turbo host-GPU mode over built binaries. See census.hpp and docs/perf-design/data/turbo-census/README.md.
//
//   turbo_census [--host SM]... [--profile SM]... [--out FILE] FILE...
//
// FILE: an ELF executable or shared library (its .nv_fatbin section is read), or a raw dump of that section. One module per fatbin container (what a
// __cudaRegisterFatBinary call registers). Defaults: hosts 75 86 89 90, simulated profiles 86 (nvidia/a10g) and 90 (nvidia/h100).
// Output, tab separated, one line per module and per kernel:
//   M <file> <module> <images> <sim86> <sim90>             images: "SASS sm_75, sm_86, PTX sm_86" (hostgpu::describe_candidates); simN: what the simulator runs for profile N
//   K <file> <module> <kernel> <cell>...                   one cell per (profile, host), in the order of the header line
//   cell: <image>:<ok|ok+unlisted|blocker,blocker...>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "census.hpp"
#include "fatbin.hpp"
#include "vgpu/sass/cubin.hpp"

using namespace vgpu;

namespace {

struct Map {
  const uint8_t* p = nullptr;
  size_t n = 0;
  ~Map() {
    if (p) munmap(const_cast<uint8_t*>(p), n);
  }
};

bool map_file(const char* path, Map* m) {
  const int fd = open(path, O_RDONLY);
  if (fd < 0) return false;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    return false;
  }
  void* a = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (a == MAP_FAILED) return false;
  m->p = static_cast<const uint8_t*>(a);
  m->n = static_cast<size_t>(st.st_size);
  return true;
}

template <class T>
T rd(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

// The .nv_fatbin sections of an ELF64 file: (offset, size).
std::vector<std::pair<size_t, size_t>> fatbin_sections(const Map& m) {
  std::vector<std::pair<size_t, size_t>> out;
  if (m.n < 64 || std::memcmp(m.p, "\x7f" "ELF", 4) != 0 || m.p[4] != 2) return out;
  const uint64_t shoff = rd<uint64_t>(m.p + 0x28);
  const uint16_t shentsize = rd<uint16_t>(m.p + 0x3a), shnum = rd<uint16_t>(m.p + 0x3c), shstrndx = rd<uint16_t>(m.p + 0x3e);
  if (!shoff || shentsize < 64 || shstrndx >= shnum || shoff + uint64_t{shnum} * shentsize > m.n) return out;
  const uint8_t* strsh = m.p + shoff + uint64_t{shstrndx} * shentsize;
  const uint64_t stroff = rd<uint64_t>(strsh + 24), strsz = rd<uint64_t>(strsh + 32);
  if (stroff + strsz > m.n) return out;
  for (unsigned i = 0; i < shnum; ++i) {
    const uint8_t* sh = m.p + shoff + uint64_t{i} * shentsize;
    const uint32_t name = rd<uint32_t>(sh);
    if (name >= strsz) continue;
    const char* nm = reinterpret_cast<const char*>(m.p + stroff + name);
    if (std::strncmp(nm, ".nv_fatbin", strsz - name < 11 ? strsz - name : 11) != 0 || std::strlen(nm) != 10) continue;
    const uint64_t off = rd<uint64_t>(sh + 24), size = rd<uint64_t>(sh + 32);
    if (off + size <= m.n) out.emplace_back(off, size);
  }
  return out;
}

// The containers of a fatbin section: a header { u32 magic 0xBA55ED50, u16 version, u16 header_size, u64 payload size } and its entries.
std::vector<std::pair<size_t, size_t>> containers(const uint8_t* p, size_t n) {
  std::vector<std::pair<size_t, size_t>> out;
  size_t at = 0;
  while (at + 16 <= n) {
    if (rd<uint32_t>(p + at) != 0xBA55ED50u) {
      at += 8;
      continue;
    }
    const size_t total = rd<uint16_t>(p + at + 6) + static_cast<size_t>(rd<uint64_t>(p + at + 8));
    if (total < 16 || at + total > n) break;
    out.emplace_back(at, total);
    at += total;
  }
  return out;
}

std::string cell_text(const census::KernelCell& c) {
  std::string s = census::image_name(c.image);
  s += ':';
  if (c.admitted) return s + (c.unlisted ? "ok+unlisted" : "ok");
  for (size_t i = 0; i < c.blockers.size(); ++i) s += (i ? "," : "") + c.blockers[i];
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  setenv("VGPU_HOST_GPU", "turbo", 1);   // (hostgpu::turbo_selected: the shim keeps every image of a fatbin only then)
  std::vector<int> hosts, profiles;
  std::vector<const char*> files;
  FILE* out = stdout;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--host" && i + 1 < argc) hosts.push_back(std::atoi(argv[++i]));
    else if (a == "--profile" && i + 1 < argc) profiles.push_back(std::atoi(argv[++i]));
    else if (a == "--out" && i + 1 < argc) out = std::fopen(argv[++i], "w");
    else files.push_back(argv[i]);
  }
  if (!out) return 2;
  if (hosts.empty()) hosts = {75, 86, 89, 90};
  if (profiles.empty()) profiles = {86, 90};
  std::fprintf(out, "H\tcells");
  for (int p : profiles)
    for (int h : hosts) std::fprintf(out, "\tp%d:h%d", p, h);
  std::fprintf(out, "\n");
  unsigned bad = 0;
  for (const char* path : files) {
    Map m;
    if (!map_file(path, &m)) {
      std::fprintf(out, "E\t%s\tcannot read\n", path);
      continue;
    }
    std::vector<std::pair<size_t, size_t>> secs = fatbin_sections(m);
    if (secs.empty() && m.n >= 4 && rd<uint32_t>(m.p) == 0xBA55ED50u) secs.emplace_back(0, m.n);   // a raw dump
    if (secs.empty()) {
      std::fprintf(out, "E\t%s\tno .nv_fatbin\n", path);
      continue;
    }
    unsigned module = 0;
    for (const auto& [soff, ssize] : secs)
      for (const auto& [coff, ctotal] : containers(m.p + soff, ssize)) {
        const uint8_t* fb = m.p + soff + coff;
        std::shared_ptr<const hostgpu::CardSource> src;
        try {
          src = cuda::card_source_from_fatbin(fb);   // (one container: the reader stops at its declared end)
        } catch (const std::exception&) {
        }
        ++module;
        if (!src) {
          std::fprintf(out, "E\t%s\t%u\tno images\n", path, module);
          ++bad;
          continue;
        }
        const std::vector<hostgpu::CardCandidate>& c = src->candidates;
        census::Analysis an;
        const std::vector<std::string> kernels = census::kernel_names(c, an);
        std::string sims;
        std::vector<int> sim_index;
        for (int p : profiles) {   // the image the simulator runs for the profile: fatbin.cpp pick_cubin, found among the candidates by its bytes
          const std::string pick = cuda::pick_cubin(fb, static_cast<uint32_t>(p));
          int idx = -1;
          for (size_t k = 0; k < c.size() && !pick.empty(); ++k)
            if (c[k].kind == hostgpu::CardCandidate::Cubin && c[k].bytes == pick) idx = static_cast<int>(k);
          sim_index.push_back(idx);
          sims += "\t" + (idx >= 0 ? "sm_" + std::to_string(c[static_cast<size_t>(idx)].arch) + (c[static_cast<size_t>(idx)].suffix ? std::string(1, c[static_cast<size_t>(idx)].suffix) : "") : std::string("PTX"));
        }
        std::fprintf(out, "M\t%s\t%u\t%s%s\n", path, module, hostgpu::describe_candidates(c).c_str(), sims.c_str());
        std::vector<std::vector<census::KernelCell>> cells;
        for (size_t pi = 0; pi < profiles.size(); ++pi)
          for (int h : hosts) cells.push_back(census::classify_module(c, sim_index[pi], h, kernels, an));
        for (size_t k = 0; k < kernels.size(); ++k) {
          std::fprintf(out, "K\t%s\t%u\t%s", path, module, kernels[k].c_str());
          for (const auto& cc : cells) std::fprintf(out, "\t%s", cell_text(cc[k]).c_str());
          std::fprintf(out, "\n");
        }
      }
  }
  if (out != stdout) std::fclose(out);
  return 0;
}
