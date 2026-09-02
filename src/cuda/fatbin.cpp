#include "fatbin.hpp"

#include <dlfcn.h>

#include <cstring>

#include "vgpu/error.hpp"

namespace vgpu::cuda {
namespace {

constexpr uint32_t kWrapperMagic = 0x466243B1;
constexpr uint32_t kFatbinMagic = 0xBA55ED50;
constexpr uint64_t kFlagLz4 = 0x2000;
constexpr uint64_t kFlagZstd = 0x8000;

struct ContainerHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t size;
};

struct EntryHeader {
  uint16_t kind;
  uint16_t version;
  uint32_t header_size;
  uint64_t padded_payload_size;
  uint32_t payload_size;  // unpadded (compressed size when compressed)
  uint32_t unknown0;
  uint16_t minor;
  uint16_t major;
  uint32_t arch;
  uint32_t name_offset;
  uint32_t name_size;
  uint64_t flags;
};
static_assert(sizeof(EntryHeader) == 48, "entry header prefix layout");

// zstd via dlopen: no build-time dependency; libzstd.so.1 ships with every
// mainstream distro (apt itself links it).
struct Zstd {
  size_t (*decompress)(void*, size_t, const void*, size_t) = nullptr;
  unsigned long long (*content_size)(const void*, size_t) = nullptr;
  unsigned (*is_error)(size_t) = nullptr;
};

const Zstd* zstd() {
  static Zstd z;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void* h = dlopen("libzstd.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (h) {
      z.decompress = reinterpret_cast<size_t (*)(void*, size_t, const void*, size_t)>(
          dlsym(h, "ZSTD_decompress"));
      z.content_size = reinterpret_cast<unsigned long long (*)(const void*, size_t)>(
          dlsym(h, "ZSTD_getFrameContentSize"));
      z.is_error = reinterpret_cast<unsigned (*)(size_t)>(dlsym(h, "ZSTD_isError"));
    }
  }
  return (z.decompress && z.content_size && z.is_error) ? &z : nullptr;
}

// The LZ4 block format, which is what nvcc used before it moved to zstd -- so
// every CUDA 12 toolkit produces it and any host with one needs this path.
// Implemented here rather than dlopened: it is forty lines, the format is
// frozen, and a fatbin is untrusted input that deserves bounds checks written
// on purpose rather than inherited.
//
// A block is a sequence of: a token byte (high nibble = literal length, low
// nibble = match length - 4), optional length-extension bytes (255 means
// "keep reading"), the literals themselves, then a little-endian 16-bit
// backward offset into the output produced so far. Matches may overlap the
// current output position, so the copy has to be byte at a time.
std::string decompress_lz4(const uint8_t* src, size_t src_size, size_t expected) {
  std::string out;
  out.reserve(expected);
  size_t ip = 0;
  auto need = [&](size_t n) {
    if (src_size - ip < n)
      throw Error::make(Err::InvalidValue, "fatbin LZ4 block ends mid-sequence");
  };
  auto read_length = [&](size_t base) {
    size_t len = base;
    if (base == 15) {
      for (;;) {
        need(1);
        const uint8_t b = src[ip++];
        len += b;
        if (b != 255) break;
        if (len > expected)
          throw Error::make(Err::InvalidValue, "fatbin LZ4 length runs past the declared size");
      }
    }
    return len;
  };

  while (ip < src_size) {
    const uint8_t token = src[ip++];
    const size_t literals = read_length(token >> 4);
    need(literals);
    if (out.size() + literals > expected)
      throw Error::make(Err::InvalidValue, "fatbin LZ4 literals exceed the declared size");
    out.append(reinterpret_cast<const char*>(src + ip), literals);
    ip += literals;
    if (ip >= src_size) break;   // the last sequence is literals only

    need(2);
    const size_t offset = static_cast<size_t>(src[ip]) | (static_cast<size_t>(src[ip + 1]) << 8);
    ip += 2;
    if (offset == 0 || offset > out.size())
      throw Error::make(Err::InvalidValue, "fatbin LZ4 match offset points outside the output");
    const size_t match = read_length(token & 0xF) + 4;   // minimum match is 4 bytes
    if (out.size() + match > expected)
      throw Error::make(Err::InvalidValue, "fatbin LZ4 match exceeds the declared size");
    const size_t start = out.size() - offset;
    for (size_t i = 0; i < match; ++i) out.push_back(out[start + i]);
  }
  if (out.size() != expected)
    throw Error::make(Err::InvalidValue, "fatbin LZ4 produced ", out.size(), " bytes, header said ",
                      expected);
  return out;
}

std::string decompress_zstd(const uint8_t* src, size_t src_size) {
  const Zstd* z = zstd();
  if (!z)
    throw Error::make(Err::Unsupported,
                      "this fatbin's PTX is zstd-compressed and libzstd.so.1 could not be loaded; "
                      "install zstd or rebuild the application with -Xfatbin=-compress=false");
  unsigned long long out_size = z->content_size(src, src_size);
  if (out_size == ~0ull || out_size == static_cast<unsigned long long>(-2) || out_size == 0 ||
      out_size > (1ull << 31))
    throw Error::make(Err::InvalidValue, "fatbin zstd frame has no valid content size");
  std::string out(out_size, '\0');
  size_t got = z->decompress(out.data(), out.size(), src, src_size);
  if (z->is_error(got) || got != out_size)
    throw Error::make(Err::InvalidValue, "fatbin zstd decompression failed");
  return out;
}

}  // namespace

std::vector<FatbinPtx> extract_ptx(const void* data) {
  if (!data) throw Error::make(Err::InvalidValue, "NULL fatbin image");
  const uint8_t* p = static_cast<const uint8_t*>(data);

  uint32_t magic = 0;
  std::memcpy(&magic, p, 4);
  if (magic == kWrapperMagic) {
    // __fatBinC_Wrapper_t { int magic; int version; const ull* data; void* filename_or_fatbins; }
    const uint8_t* inner = nullptr;
    std::memcpy(&inner, p + 8, 8);
    if (!inner) throw Error::make(Err::InvalidValue, "fatbin wrapper has NULL data pointer");
    p = inner;
    std::memcpy(&magic, p, 4);
  }
  if (magic != kFatbinMagic)
    throw Error::make(Err::InvalidValue, "not a fatbin image (bad magic; expected 0xBA55ED50)");

  ContainerHeader ch{};
  std::memcpy(&ch, p, sizeof ch);
  if (ch.header_size < sizeof(ContainerHeader))
    throw Error::make(Err::InvalidValue, "fatbin container header_size is ", ch.header_size,
                      ", smaller than the header itself");
  if (ch.size > kMaxFatbinBytes)
    throw Error::make(Err::InvalidValue, "fatbin declares ", ch.size,
                      " bytes, above the ", kMaxFatbinBytes, "-byte limit; image looks corrupt");

  std::vector<FatbinPtx> out;
  const uint8_t* const body = p + ch.header_size;
  const uint8_t* const end = body + ch.size;
  const uint8_t* e = body;
  // Every iteration must advance strictly, so a zero-length entry cannot spin.
  while (e < end) {
    if (static_cast<uint64_t>(end - e) < sizeof(EntryHeader)) break;  // trailing padding
    EntryHeader eh{};
    std::memcpy(&eh, e, sizeof eh);
    if (eh.header_size < sizeof(EntryHeader))
      throw Error::make(Err::InvalidValue, "fatbin entry header_size is ", eh.header_size,
                        ", smaller than the entry header");
    if (static_cast<uint64_t>(end - e) < eh.header_size)
      throw Error::make(Err::InvalidValue, "fatbin entry header runs past the end of the image");
    const uint8_t* payload = e + eh.header_size;
    if (eh.padded_payload_size > static_cast<uint64_t>(end - payload))
      throw Error::make(Err::InvalidValue, "fatbin entry payload (", eh.padded_payload_size,
                        " bytes) runs past the end of the image");

    if (eh.kind == 1 /* PTX */) {
      uint64_t size = eh.payload_size ? eh.payload_size : eh.padded_payload_size;
      if (size > eh.padded_payload_size)
        throw Error::make(Err::InvalidValue, "fatbin PTX payload_size (", size,
                          ") exceeds its padded size (", eh.padded_payload_size, ")");
      FatbinPtx px;
      px.arch = eh.arch;
      if (eh.flags & kFlagZstd) {
        px.text = decompress_zstd(payload, static_cast<size_t>(size));
      } else if (eh.flags & kFlagLz4) {
        // A compressed entry carries the uncompressed size in an extended
        // header; without it there is nothing to size the output against.
        if (eh.header_size < 64)
          throw Error::make(Err::InvalidValue,
                            "fatbin entry is LZ4-compressed but its header is too short to carry "
                            "the uncompressed size");
        uint64_t uncompressed = 0;
        std::memcpy(&uncompressed, e + 56, sizeof uncompressed);
        if (uncompressed == 0 || uncompressed > kMaxFatbinBytes)
          throw Error::make(Err::InvalidValue, "fatbin LZ4 entry declares an uncompressed size of ",
                            uncompressed, ", which is not usable");
        px.text = decompress_lz4(payload, static_cast<size_t>(size),
                                 static_cast<size_t>(uncompressed));
      } else {
        px.text.assign(reinterpret_cast<const char*>(payload), static_cast<size_t>(size));
      }
      while (!px.text.empty() && px.text.back() == '\0') px.text.pop_back();
      out.push_back(std::move(px));
    }

    const uint8_t* next = payload + eh.padded_payload_size;
    if (next <= e)
      throw Error::make(Err::InvalidValue,
                        "fatbin entry does not advance (header_size and payload both zero); "
                        "image is malformed");
    e = next;
  }
  return out;
}

}  // namespace vgpu::cuda
