// Clang offload bundles (vgpu/amd_bundle.hpp).
//
// The layouts are clang's (clang/lib/Driver/OffloadBundler.cpp): a bundle is
// "__CLANG_OFFLOAD_BUNDLE__", a count, and for each entry the offset, size
// and target triple of its code object. A compressed one is "CCOB", a format
// version, the compression method, sizes and a hash, then the compressed
// bundle. Inflating it takes zstd or zlib, which are not built in: the
// system's are loaded when a compressed bundle first turns up, so VirtualGPU
// builds with nothing beyond the compiler, and runs a library that ships
// compressed code wherever the library itself would run.
#include "vgpu/amd_bundle.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <vector>
#include <mutex>

#include "vgpu/error.hpp"

namespace vgpu::amd {

namespace {

constexpr char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr char kCompressedMagic[] = "CCOB";

template <typename T>
T at(const uint8_t* b, size_t offset) {
  T v;
  std::memcpy(&v, b + offset, sizeof v);
  return v;
}

// One of the two libraries, loaded once, and the one function taken from it.
template <typename Fn>
Fn library_function(const char* soname, const char* name) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  void* lib = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
  if (!lib)
    throw Error::make(Err::Unsupported, "the device code is compressed, and inflating it needs ", soname,
                      ", which is not installed");
  void* fn = dlsym(lib, name);
  if (!fn) throw Error::make(Err::Unsupported, soname, " has no ", name);
  return reinterpret_cast<Fn>(fn);
}

// A bundle's table of contents: each entry's triple, and where its code is.
struct Entry {
  std::string triple;
  uint64_t offset = 0, size = 0;
};
// Reads the table from the start of an uncompressed bundle of which `have`
// bytes are known; false where they do not yet reach the table's end.
bool read_entries(const uint8_t* b, uint64_t have, std::vector<Entry>* out) {
  if (have < 32) return false;
  const uint64_t count = at<uint64_t>(b, 24);
  uint64_t entry = 32;
  out->clear();
  for (uint64_t i = 0; i < count; ++i) {
    if (entry + 24 > have) return false;
    Entry e;
    e.offset = at<uint64_t>(b, entry);
    e.size = at<uint64_t>(b, entry + 8);
    const uint64_t triple_len = at<uint64_t>(b, entry + 16);
    if (entry + 24 + triple_len > have) return false;
    e.triple.assign(reinterpret_cast<const char*>(b + entry + 24), triple_len);
    entry += 24 + triple_len;
    out->push_back(std::move(e));
  }
  return true;
}

// "hipv4-amdgcn-amd-amdhsa--gfx942", perhaps with ":sramecc+:xnack-" after it:
// the target and its features are what follow the last "--". The host's
// entry, and anything that is not HIP device code, is empty.
std::string target_of(const Entry& e) {
  const size_t dashes = e.triple.rfind("--");
  if (e.triple.rfind("hip", 0) != 0 || dashes == std::string::npos || !e.size) return "";
  return e.triple.substr(dashes + 2);
}
std::string processor(const std::string& target) { return target.substr(0, target.find(':')); }

// Whether a bundle read for `device` keeps this entry's code.
bool kept(const std::string& target, const std::string& device) {
  return !target.empty() && (device.empty() || processor(target) == processor(device));
}

// The compressed header, by format version: the magic, the version and the
// method, then the whole thing's size (not in version 1), the size inflated,
// and a hash of it.
struct Compressed {
  uint16_t method = 0;
  const uint8_t* data = nullptr;
  uint64_t size = 0, out_size = 0;
};
Compressed compressed(const uint8_t* b) {
  const uint16_t version = at<uint16_t>(b, 4);
  Compressed c;
  c.method = at<uint16_t>(b, 6);
  uint64_t total = 0, header = 0;
  if (version == 2) {
    total = at<uint32_t>(b, 8), c.out_size = at<uint32_t>(b, 12), header = 24;
  } else if (version == 3) {
    total = at<uint64_t>(b, 8), c.out_size = at<uint64_t>(b, 16), header = 32;
  } else {
    // Version 1 does not say how long it is, which a bundle in memory needs.
    throw Error::make(Err::Unsupported, "a compressed offload bundle of format version ", version,
                      ", which this does not read");
  }
  if (total < header) throw Error::make(Err::InvalidValue, "a compressed offload bundle shorter than its header");
  if (c.method > 1)
    throw Error::make(Err::Unsupported, "a compressed offload bundle whose method (", c.method, ") is not zlib or zstd");
  c.data = b + header;
  c.size = total - header;
  return c;
}

// Where each kept entry's code goes in Bundle::inflated, and how much that is.
struct Plan {
  std::vector<Entry> entries;
  std::vector<uint64_t> placed;   // per entry: its offset in `inflated`, or ~0 where it is not kept
  uint64_t bytes = 0;
};
Plan plan(std::vector<Entry> entries, const std::string& device, uint64_t out_size) {
  Plan p;
  p.entries = std::move(entries);
  for (const Entry& e : p.entries) {
    if (!kept(target_of(e), device)) {
      p.placed.push_back(~uint64_t{0});
      continue;
    }
    if (e.offset + e.size > out_size)
      throw Error::make(Err::InvalidValue, "an offload bundle entry (", e.triple, ") runs past the bundle's end");
    p.placed.push_back(p.bytes);
    p.bytes += e.size;
  }
  return p;
}
// Copies what of the inflated bytes [pos, pos + n) the plan keeps.
void keep(const Plan& p, uint64_t pos, const uint8_t* bytes, uint64_t n, std::string* into) {
  for (size_t i = 0; i < p.entries.size(); ++i) {
    if (p.placed[i] == ~uint64_t{0}) continue;
    const Entry& e = p.entries[i];
    const uint64_t from = std::max(pos, e.offset), to = std::min(pos + n, e.offset + e.size);
    if (from < to) std::memcpy(into->data() + p.placed[i] + (from - e.offset), bytes + (from - pos), to - from);
  }
}

// zstd's streaming calls, as its documentation gives them: the inflated bundle
// passes through a buffer of a few megabytes, and what is kept is copied out.
struct ZstdIn {
  const void* src;
  size_t size, pos;
};
struct ZstdOut {
  void* dst;
  size_t size, pos;
};
void inflate_zstd(const Compressed& c, const std::string& device, Plan* p, std::string* into) {
  using Create = void* (*)();
  using Free = size_t (*)(void*);
  using Stream = size_t (*)(void*, ZstdOut*, ZstdIn*);
  using IsError = unsigned (*)(size_t);
  using ErrorName = const char* (*)(size_t);
  static const auto create = library_function<Create>("libzstd.so.1", "ZSTD_createDStream");
  static const auto destroy = library_function<Free>("libzstd.so.1", "ZSTD_freeDStream");
  static const auto stream = library_function<Stream>("libzstd.so.1", "ZSTD_decompressStream");
  static const auto is_error = library_function<IsError>("libzstd.so.1", "ZSTD_isError");
  static const auto error_name = library_function<ErrorName>("libzstd.so.1", "ZSTD_getErrorName");
  std::unique_ptr<void, Free> ds(create(), destroy);
  if (!ds) throw Error::make(Err::Unsupported, "zstd could not make a decompression stream");
  std::string head;   // the start, until the table of contents is in it
  bool planned = false;
  std::vector<uint8_t> buffer(size_t{4} << 20);
  ZstdIn in{c.data, c.size, 0};
  uint64_t pos = 0;
  for (;;) {
    ZstdOut out{buffer.data(), buffer.size(), 0};
    const size_t r = stream(ds.get(), &out, &in);
    if (is_error(r)) throw Error::make(Err::InvalidValue, "the compressed device code does not inflate: ", error_name(r));
    if (pos + out.pos > c.out_size)
      throw Error::make(Err::InvalidValue, "the compressed device code inflates to more than the ", c.out_size,
                        " bytes its header says");
    if (!planned) {
      head.append(reinterpret_cast<const char*>(buffer.data()), out.pos);
      std::vector<Entry> entries;
      if (std::memcmp(head.data(), kMagic, std::min<size_t>(head.size(), 24)) != 0)
        throw Error::make(Err::InvalidValue, "the compressed device code is not an offload bundle");
      if (read_entries(reinterpret_cast<const uint8_t*>(head.data()), head.size(), &entries)) {
        *p = plan(std::move(entries), device, c.out_size);
        into->assign(p->bytes, '\0');
        keep(*p, 0, reinterpret_cast<const uint8_t*>(head.data()), head.size(), into);
        planned = true;
        std::string().swap(head);
      }
    } else {
      keep(*p, pos, buffer.data(), out.pos, into);
    }
    pos += out.pos;
    if (r == 0 && in.pos == in.size) break;   // a frame finished, and nothing is left
    if (in.pos == in.size && out.pos == 0)
      throw Error::make(Err::InvalidValue, "the compressed device code ends before it is whole");
  }
  if (pos != c.out_size)
    throw Error::make(Err::InvalidValue, "the compressed device code inflates to ", pos, " bytes, not the ",
                      c.out_size, " its header says");
  if (!planned) throw Error::make(Err::InvalidValue, "the compressed device code is not an offload bundle");
}

// zlib's one-shot call: the whole bundle, then what is kept of it. zlib is
// what clang compressed with before it took zstd.
void inflate_zlib(const Compressed& c, const std::string& device, Plan* p, std::string* into) {
  using Uncompress = int (*)(unsigned char*, unsigned long*, const unsigned char*, unsigned long);
  static const auto uncompress = library_function<Uncompress>("libz.so.1", "uncompress");
  std::string all(c.out_size, '\0');
  unsigned long n = c.out_size;
  if (const int e = uncompress(reinterpret_cast<unsigned char*>(all.data()), &n, c.data, c.size); e != 0)
    throw Error::make(Err::InvalidValue, "the compressed device code does not inflate (zlib says ", e, ")");
  if (n != c.out_size)
    throw Error::make(Err::InvalidValue, "the compressed device code inflates to ", n, " bytes, not the ",
                      c.out_size, " its header says");
  std::vector<Entry> entries;
  const uint8_t* b = reinterpret_cast<const uint8_t*>(all.data());
  if (n < 24 || std::memcmp(b, kMagic, 24) != 0 || !read_entries(b, n, &entries))
    throw Error::make(Err::InvalidValue, "the compressed device code is not an offload bundle");
  *p = plan(std::move(entries), device, c.out_size);
  into->assign(p->bytes, '\0');
  keep(*p, 0, b, n, into);
}

}  // namespace

bool is_bundle(const uint8_t* b) {
  return std::memcmp(b, kMagic, 24) == 0 || std::memcmp(b, kCompressedMagic, 4) == 0;
}

std::unique_ptr<Bundle> read_bundle(const uint8_t* b, const std::string& device) {
  auto out = std::make_unique<Bundle>();
  if (std::memcmp(b, kCompressedMagic, 4) == 0) {
    const Compressed c = compressed(b);
    Plan p;
    if (c.method == 1) inflate_zstd(c, device, &p, &out->inflated);
    else inflate_zlib(c, device, &p, &out->inflated);
    for (size_t i = 0; i < p.entries.size(); ++i) {
      const std::string target = target_of(p.entries[i]);
      if (target.empty()) continue;
      out->all.push_back(target);
      if (p.placed[i] != ~uint64_t{0})
        out->targets[target] = std::string_view(out->inflated.data() + p.placed[i], p.entries[i].size);
    }
    return out;
  }
  if (std::memcmp(b, kMagic, 24) != 0) return nullptr;
  std::vector<Entry> entries;
  read_entries(b, ~uint64_t{0}, &entries);   // in memory already, however long it is
  for (const Entry& e : entries) {
    const std::string target = target_of(e);
    if (target.empty()) continue;
    out->all.push_back(target);
    if (kept(target, device))
      out->targets[target] = std::string_view(reinterpret_cast<const char*>(b + e.offset), e.size);
  }
  return out;
}

const std::string_view* code_for(const Bundle& bundle, const std::string& device) {
  const std::string bare = device.substr(0, device.find(':'));
  const std::string_view* best = nullptr;
  size_t named = 0;
  for (const auto& [target, code] : bundle.targets) {
    if (target.substr(0, target.find(':')) != bare) continue;
    bool fits = true;
    size_t n = 0;
    for (size_t colon = target.find(':'); colon != std::string::npos; colon = target.find(':', colon + 1), ++n) {
      const std::string feature = target.substr(colon, target.find(':', colon + 1) - colon);   // ":xnack-"
      if ((device + ":").find(feature + ":") == std::string::npos) fits = false;
    }
    if (fits && (!best || n > named)) {
      best = &code;
      named = n;
    }
  }
  return best;
}

std::string target_list(const Bundle& bundle) {
  std::vector<std::string> all = bundle.all;
  std::sort(all.begin(), all.end());
  all.erase(std::unique(all.begin(), all.end()), all.end());
  std::string list;
  for (const auto& t : all) list += (list.empty() ? "" : ", ") + t;
  return list.empty() ? "no GPU" : list;
}

}  // namespace vgpu::amd
