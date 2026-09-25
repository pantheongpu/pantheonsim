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

#include <cstring>
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

// zstd's and zlib's one-shot calls, as their documentation gives them.
std::string inflate_zstd(const uint8_t* src, size_t size, size_t out_size) {
  using Decompress = size_t (*)(void*, size_t, const void*, size_t);
  using IsError = unsigned (*)(size_t);
  using ErrorName = const char* (*)(size_t);
  static const auto decompress = library_function<Decompress>("libzstd.so.1", "ZSTD_decompress");
  static const auto is_error = library_function<IsError>("libzstd.so.1", "ZSTD_isError");
  static const auto error_name = library_function<ErrorName>("libzstd.so.1", "ZSTD_getErrorName");
  std::string out(out_size, '\0');
  const size_t n = decompress(out.data(), out.size(), src, size);
  if (is_error(n)) throw Error::make(Err::InvalidValue, "the compressed device code does not inflate: ", error_name(n));
  if (n != out_size)
    throw Error::make(Err::InvalidValue, "the compressed device code inflates to ", n, " bytes, not the ", out_size,
                      " its header says");
  return out;
}

std::string inflate_zlib(const uint8_t* src, size_t size, size_t out_size) {
  using Uncompress = int (*)(unsigned char*, unsigned long*, const unsigned char*, unsigned long);
  static const auto uncompress = library_function<Uncompress>("libz.so.1", "uncompress");
  std::string out(out_size, '\0');
  unsigned long n = out_size;
  if (const int e = uncompress(reinterpret_cast<unsigned char*>(out.data()), &n, src, size); e != 0)
    throw Error::make(Err::InvalidValue, "the compressed device code does not inflate (zlib says ", e, ")");
  if (n != out_size)
    throw Error::make(Err::InvalidValue, "the compressed device code inflates to ", n, " bytes, not the ", out_size,
                      " its header says");
  return out;
}

// The header of a compressed bundle, by format version: the magic, the
// version and the method, then the whole thing's size (not in version 1), the
// size inflated, and a hash of it.
std::string inflate(const uint8_t* b) {
  const uint16_t version = at<uint16_t>(b, 4), method = at<uint16_t>(b, 6);
  uint64_t total = 0, out_size = 0, header = 0;
  if (version == 2) {
    total = at<uint32_t>(b, 8), out_size = at<uint32_t>(b, 12), header = 24;
  } else if (version == 3) {
    total = at<uint64_t>(b, 8), out_size = at<uint64_t>(b, 16), header = 32;
  } else {
    // Version 1 does not say how long it is, which a bundle in memory needs.
    throw Error::make(Err::Unsupported, "a compressed offload bundle of format version ", version,
                      ", which this does not read");
  }
  if (total < header) throw Error::make(Err::InvalidValue, "a compressed offload bundle shorter than its header");
  switch (method) {
    case 0: return inflate_zlib(b + header, total - header, out_size);
    case 1: return inflate_zstd(b + header, total - header, out_size);
    default: throw Error::make(Err::Unsupported, "a compressed offload bundle whose method (", method, ") is not zlib or zstd");
  }
}

}  // namespace

bool is_bundle(const uint8_t* b) {
  return std::memcmp(b, kMagic, 24) == 0 || std::memcmp(b, kCompressedMagic, 4) == 0;
}

std::unique_ptr<Bundle> read_bundle(const uint8_t* b) {
  auto out = std::make_unique<Bundle>();
  if (std::memcmp(b, kCompressedMagic, 4) == 0) {
    out->inflated = inflate(b);
    b = reinterpret_cast<const uint8_t*>(out->inflated.data());
  }
  if (std::memcmp(b, kMagic, 24) != 0) return nullptr;
  const uint64_t count = at<uint64_t>(b, 24);
  uint64_t entry = 32;
  for (uint64_t i = 0; i < count; ++i) {
    const uint64_t offset = at<uint64_t>(b, entry), size = at<uint64_t>(b, entry + 8);
    const uint64_t triple_len = at<uint64_t>(b, entry + 16);
    const std::string triple(reinterpret_cast<const char*>(b + entry + 24), triple_len);
    entry += 24 + triple_len;
    if (!out->inflated.empty() && offset + size > out->inflated.size())
      throw Error::make(Err::InvalidValue, "an offload bundle entry (", triple, ") runs past the bundle's end");
    // "hipv4-amdgcn-amd-amdhsa--gfx942", perhaps with ":sramecc+:xnack-" after
    // it: the target and its features are what follow the last "--". The
    // host's entry, and anything that is not HIP device code, is passed over.
    const size_t dashes = triple.rfind("--");
    if (triple.rfind("hip", 0) != 0 || dashes == std::string::npos || !size) continue;
    out->targets[triple.substr(dashes + 2)] = std::string_view(reinterpret_cast<const char*>(b + offset), size);
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
  std::string list;
  for (const auto& t : bundle.targets) list += (list.empty() ? "" : ", ") + t.first;
  return list.empty() ? "no GPU" : list;
}

}  // namespace vgpu::amd
