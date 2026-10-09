#include "fatbin.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

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
  size_t (*compress)(void*, size_t, const void*, size_t, int) = nullptr;
  size_t (*bound)(size_t) = nullptr;
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
      z.compress = reinterpret_cast<size_t (*)(void*, size_t, const void*, size_t, int)>(
          dlsym(h, "ZSTD_compress"));
      z.bound = reinterpret_cast<size_t (*)(size_t)>(dlsym(h, "ZSTD_compressBound"));
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

std::string compress_zstd(const std::string& data, bool high) {
  const Zstd* z = zstd();
  if (!z || !z->compress || !z->bound || data.empty()) return {};
  std::string out(z->bound(data.size()), '\0');
  const size_t n = z->compress(out.data(), out.size(), data.data(), data.size(), high ? 19 : 3);
  if (z->is_error(n)) return {};
  out.resize(n);
  return out;
}

// The LZ4 block format (see decompress_lz4), written greedily with a hash of four bytes. The format's
// end rules are kept, so any LZ4 decoder takes it: the last match starts at least 12 bytes before
// the end, and the last 5 bytes are literals.
std::string compress_lz4(const std::string& data) {
  const auto* src = reinterpret_cast<const uint8_t*>(data.data());
  const size_t n = data.size();
  std::string out;
  auto length_ext = [&](size_t len) {   // what a nibble of 15 leaves to say
    while (len >= 255) {
      out.push_back(static_cast<char>(255));
      len -= 255;
    }
    out.push_back(static_cast<char>(len));
  };
  auto emit = [&](size_t lit_at, size_t lit_len, size_t offset, size_t match_len) {
    const size_t token_at = out.size();
    out.push_back(0);
    uint8_t token = static_cast<uint8_t>((lit_len >= 15 ? 15 : lit_len) << 4);
    if (lit_len >= 15) length_ext(lit_len - 15);
    out.append(data, lit_at, lit_len);
    if (match_len) {
      out.push_back(static_cast<char>(offset & 0xff));
      out.push_back(static_cast<char>(offset >> 8));
      const size_t m = match_len - 4;
      token |= static_cast<uint8_t>(m >= 15 ? 15 : m);
      if (m >= 15) length_ext(m - 15);
    }
    out[token_at] = static_cast<char>(token);
  };
  constexpr size_t kHashBits = 14;
  std::vector<uint32_t> table(size_t{1} << kHashBits, UINT32_MAX);
  size_t anchor = 0, i = 0;
  if (n >= 13) {
    const size_t last_match_start = n - 12;   // a match may not start later (the format's rule)
    const size_t match_limit = n - 5;         // nor reach into the last 5 bytes
    while (i < last_match_start) {
      uint32_t v;
      std::memcpy(&v, src + i, 4);
      const uint32_t h = (v * 2654435761u) >> (32 - kHashBits);
      const uint32_t cand = table[h];
      table[h] = static_cast<uint32_t>(i);
      if (cand != UINT32_MAX && i - cand <= 65535) {
        uint32_t w;
        std::memcpy(&w, src + cand, 4);
        if (w == v) {
          size_t len = 4;
          while (i + len < match_limit && src[cand + len] == src[i + len]) ++len;
          emit(anchor, i - anchor, i - cand, len);
          i += len;
          anchor = i;
          continue;
        }
      }
      ++i;
    }
  }
  emit(anchor, n - anchor, 0, 0);   // the last literals
  return out;
}

std::string strip_ptx_comments(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  size_t i = 0;
  bool quoted = false;
  while (i < in.size()) {
    const char c = in[i];
    if (quoted) {
      out.push_back(c);
      quoted = c != '"';
      ++i;
    } else if (c == '"') {
      quoted = true;
      out.push_back(c);
      ++i;
    } else if (c == '/' && i + 1 < in.size() && in[i + 1] == '/') {
      while (i < in.size() && in[i] != '\n' && in[i] != '\r') ++i;
    } else if (c == '/' && i + 1 < in.size() && in[i + 1] == '*') {
      const size_t end = in.find("*/", i + 1);
      i = end == std::string::npos ? in.size() : end + 2;
    } else {
      out.push_back(c);
      ++i;
    }
  }
  return out;
}

std::vector<FatbinPtx> extract_ptx(const void* data) {
  return extract_ptx(data, std::numeric_limits<size_t>::max());
}

std::vector<FatbinPtx> extract_ptx(const void* data, size_t bytes) {
  std::vector<FatbinPtx> out;
  for (FatbinImage& im : extract_images(data, bytes, nullptr, kFatbinPtx | kFatbinRelocPtx)) {
    FatbinPtx px;
    px.arch = im.arch;
    px.text = std::move(im.data);
    while (!px.text.empty() && px.text.back() == '\0') px.text.pop_back();
    out.push_back(std::move(px));
  }
  return out;
}

std::vector<FatbinImage> extract_images(const void* data, size_t bytes, size_t* consumed,
                                       uint16_t kinds) {
  if (!data) throw Error::make(Err::InvalidValue, "NULL fatbin image");
  const uint8_t* p = static_cast<const uint8_t*>(data);

  // When the caller knows the buffer size, `avail` is the number of bytes
  // still inside it; when it does not, it is "unknown" and only the format's
  // internal consistency and kMaxFatbinBytes constrain the walk. Every read
  // below is checked against this rather than against a bound computed from a
  // field in the image, which is what let a declared size larger than the
  // buffer walk off the end.
  const bool bounded = bytes != std::numeric_limits<size_t>::max();
  auto room = [&](const uint8_t* q, uint64_t n) {
    if (!bounded) return true;
    const uint8_t* base = static_cast<const uint8_t*>(data);
    if (q < base) return false;
    const uint64_t used = static_cast<uint64_t>(q - base);
    return used <= bytes && n <= bytes - used;
  };
  auto need = [&](const uint8_t* q, uint64_t n, const char* what) {
    if (!room(q, n))
      throw Error::make(Err::InvalidValue, "fatbin image is truncated: ", what, " needs ", n,
                        " bytes but the buffer is only ", bytes, " long");
  };

  need(p, 4, "the magic number");
  uint32_t magic = 0;
  std::memcpy(&magic, p, 4);
  bool wrapped = false;
  if (magic == kWrapperMagic) {
    need(p, 16, "the fatbin wrapper");
    // __fatBinC_Wrapper_t { int magic; int version; const ull* data; void* filename_or_fatbins; }
    const uint8_t* inner = nullptr;
    std::memcpy(&inner, p + 8, 8);
    if (!inner) throw Error::make(Err::InvalidValue, "fatbin wrapper has NULL data pointer");
    p = inner;
    // The wrapper points at a separate allocation whose size we were not told,
    // so the caller's bound no longer applies past this point.
    bytes = std::numeric_limits<size_t>::max();
    data = inner;
    wrapped = true;
    std::memcpy(&magic, p, 4);
  }
  if (magic != kFatbinMagic)
    throw Error::make(Err::InvalidValue, "not a fatbin image (bad magic; expected 0xBA55ED50)");

  ContainerHeader ch{};
  need(p, sizeof ch, "the container header");
  std::memcpy(&ch, p, sizeof ch);
  if (ch.header_size < sizeof(ContainerHeader))
    throw Error::make(Err::InvalidValue, "fatbin container header_size is ", ch.header_size,
                      ", smaller than the header itself");
  if (ch.size > kMaxFatbinBytes)
    throw Error::make(Err::InvalidValue, "fatbin declares ", ch.size,
                      " bytes, above the ", kMaxFatbinBytes, "-byte limit; image looks corrupt");

  std::vector<FatbinImage> out;
  need(p, ch.header_size, "the container header it declares");
  const uint8_t* const body = p + ch.header_size;
  // The declared body size is a number from the image. Where the real length is
  // known, a declaration larger than the buffer is a truncated image, not a
  // licence to read that far.
  need(body, ch.size, "the container body it declares");
  const uint8_t* const end = body + ch.size;
  if (consumed) *consumed = wrapped ? 16 : static_cast<size_t>(end - p);
  const uint8_t* e = body;
  // Every iteration must advance strictly, so a zero-length entry cannot spin.
  while (e < end) {
    if (static_cast<uint64_t>(end - e) < sizeof(EntryHeader)) break;  // trailing padding
    EntryHeader eh{};
    need(e, sizeof eh, "an entry header");
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

    const uint8_t* next = payload + eh.padded_payload_size;
    if (next <= e)
      throw Error::make(Err::InvalidValue,
                        "fatbin entry does not advance (header_size and payload both zero); "
                        "image is malformed");
    // An entry the caller did not ask for is not decompressed: a SASS image
    // the PTX path never reads must not fail it (when libzstd is absent, say).
    if (!(eh.kind & kinds)) {
      e = next;
      continue;
    }
    uint64_t size = eh.payload_size ? eh.payload_size : eh.padded_payload_size;
    if (size > eh.padded_payload_size)
      throw Error::make(Err::InvalidValue, "fatbin entry payload_size (", size,
                        ") exceeds its padded size (", eh.padded_payload_size, ")");
    FatbinImage im;
    im.kind = eh.kind;
    im.arch = eh.arch;
    im.major = eh.major;
    im.minor = eh.minor;
    if (eh.name_size && eh.name_offset >= sizeof(EntryHeader) &&
        uint64_t{eh.name_offset} + eh.name_size <= eh.header_size)
      im.name.assign(reinterpret_cast<const char*>(e + eh.name_offset), eh.name_size);
    while (!im.name.empty() && im.name.back() == '\0') im.name.pop_back();
    auto decode = [&]() -> std::string {
      if (eh.flags & kFlagZstd) return decompress_zstd(payload, static_cast<size_t>(size));
      if (eh.flags & kFlagLz4) {
        // A compressed entry carries the uncompressed size in an extended
        // header; without it there is nothing to size the output against.
        if (eh.header_size < 64)
          throw Error::make(Err::InvalidValue,
                            "fatbin entry is LZ4-compressed but its header is too short to carry "
                            "the uncompressed size");
        uint64_t uncompressed = 0;
        need(e + 56, sizeof uncompressed, "an LZ4 entry's uncompressed size");
        std::memcpy(&uncompressed, e + 56, sizeof uncompressed);
        if (uncompressed == 0 || uncompressed > kMaxFatbinBytes)
          throw Error::make(Err::InvalidValue, "fatbin LZ4 entry declares an uncompressed size of ",
                            uncompressed, ", which is not usable");
        return decompress_lz4(payload, static_cast<size_t>(size), static_cast<size_t>(uncompressed));
      }
      return std::string(reinterpret_cast<const char*>(payload), static_cast<size_t>(size));
    };
    if (is_ptx_kind(eh.kind)) {
      im.stored_payload.assign(reinterpret_cast<const char*>(payload), static_cast<size_t>(size));
      // The options string the entry records: at the offset eh.unknown0 names, {string offset, size}.
      if (eh.unknown0 && uint64_t{eh.unknown0} + 8 <= eh.header_size) {
        uint32_t so = 0, sl = 0;
        std::memcpy(&so, e + eh.unknown0, 4);
        std::memcpy(&sl, e + eh.unknown0 + 4, 4);
        if (uint64_t{so} + sl <= eh.header_size) im.options.assign(reinterpret_cast<const char*>(e + so), sl);
        while (!im.options.empty() && im.options.back() == '\0') im.options.pop_back();
      }
      im.stored_flags = eh.flags;
      if (eh.header_size >= 64 && room(e + 56, 8)) std::memcpy(&im.stored_uncompressed, e + 56, 8);
      im.data = decode();
    } else {
      // Not every flagged payload is a zstd or LZ4 stream: nvcc's LTO-IR
      // entries (flags 0x18011) carry something else. What a caller needs of
      // a non-PTX entry is mostly its kind and arch, so one that does not
      // decompress is kept as stored rather than failing the whole fatbin.
      try {
        im.data = decode();
      } catch (const Error&) {
        im.data.assign(reinterpret_cast<const char*>(payload), static_cast<size_t>(size));
        im.stored = true;
      }
    }
    out.push_back(std::move(im));
    e = next;
  }
  return out;
}

std::vector<FatbinPtx> extract_elf(const void* data, size_t bytes) {
  std::vector<FatbinPtx> out;
  for (FatbinImage& im : extract_images(data, bytes, nullptr, kFatbinElf))
    if (!im.stored) out.push_back({im.arch, std::move(im.data)});
  return out;
}

std::vector<FatbinPtx> extract_elf(const void* data) {
  return extract_elf(data, std::numeric_limits<size_t>::max());
}

// The ".target sm_XY[a|f]" line of a PTX image: the number, and the suffix
// ('a', 'f' or 0). Images without one read as their header's arch, plain.
static void ptx_target(const FatbinPtx& p, uint32_t* arch, char* suffix) {
  *arch = p.arch;
  *suffix = 0;
  const size_t at = p.text.find(".target");
  if (at == std::string::npos) return;
  const size_t sm = p.text.find("sm_", at);
  const size_t eol = p.text.find('\n', at);
  if (sm == std::string::npos || (eol != std::string::npos && sm > eol)) return;
  size_t i = sm + 3;
  uint32_t n = 0;
  bool digits = false;
  while (i < p.text.size() && p.text[i] >= '0' && p.text[i] <= '9') {
    n = n * 10 + static_cast<uint32_t>(p.text[i] - '0');
    ++i;
    digits = true;
  }
  if (!digits) return;
  *arch = n;
  if (i < p.text.size() && (p.text[i] == 'a' || p.text[i] == 'f')) *suffix = p.text[i];
}

size_t pick_ptx(const std::vector<FatbinPtx>& ptxs, uint32_t cc) {
  if (ptxs.empty()) return 0;
  auto specificity = [](char s) { return s == 'a' ? 2 : s == 'f' ? 1 : 0; };
  size_t best = ptxs.size();
  uint32_t best_arch = 0;
  char best_suffix = 0;
  for (size_t i = 0; i < ptxs.size(); ++i) {
    uint32_t arch;
    char suffix;
    ptx_target(ptxs[i], &arch, &suffix);
    const bool runs = suffix == 'a'   ? arch == cc
                      : suffix == 'f' ? (arch / 10 == cc / 10 && arch <= cc)
                                      : arch <= cc;
    if (!runs) continue;
    if (best == ptxs.size() || arch > best_arch ||
        (arch == best_arch && specificity(suffix) > specificity(best_suffix))) {
      best = i;
      best_arch = arch;
      best_suffix = suffix;
    }
  }
  if (best == ptxs.size()) {
    best = 0;
    for (size_t i = 1; i < ptxs.size(); ++i)
      if (ptxs[i].arch > ptxs[best].arch) best = i;
  }
  return best;
}

/* ---- writing fatbins, and finding them in host objects ---- */

std::string write_fatbin(const std::vector<FatbinImage>& images) {
  auto put = [](std::string& out, uint64_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
  };
  auto pad8 = [](std::string& out) {
    while (out.size() % 8) out.push_back('\0');
  };
  std::string body;
  for (const FatbinImage& im : images) {
    // The header's tail: the identifier at offset 64, NUL-terminated and
    // padded to 8, then for PTX an {offset, size} record of the options
    // string that follows it. NVIDIA's writes an 8-byte empty string when
    // there are no options, and so does this.
    std::string tail;
    tail += im.name;
    tail.push_back('\0');
    pad8(tail);
    uint32_t options_record = 0;
    if (im.is_ptx() || im.kind == kFatbinLtoIr) {
      options_record = static_cast<uint32_t>(sizeof(EntryHeader) + 16 + tail.size());
      std::string opts = im.options;
      opts.push_back('\0');
      pad8(opts);
      put(tail, options_record + 8, 4);
      put(tail, im.options.size(), 4);
      tail += opts;
    }
    std::string payload = im.data;
    const bool compressed = (im.flags & (kFatbinLz4 | kFatbinZstd)) != 0;
    if (!compressed && im.is_ptx() && (payload.empty() || payload.back() != '\0')) payload.push_back('\0');
    const uint32_t payload_size = compressed ? static_cast<uint32_t>(payload.size()) : 0;
    pad8(payload);
    const uint32_t header_size = static_cast<uint32_t>(sizeof(EntryHeader) + 16 + tail.size());
    put(body, im.kind, 2);
    put(body, 0x0101, 2);                 // entry version, as both writers set it
    put(body, header_size, 4);
    put(body, payload.size(), 8);         // padded payload size
    put(body, payload_size, 4);           // payload size: 0 means "uncompressed, see above"
    put(body, options_record, 4);
    put(body, im.minor, 2);
    put(body, im.major, 2);
    put(body, im.arch, 4);
    put(body, sizeof(EntryHeader) + 16, 4);   // identifier offset
    put(body, im.name.size(), 4);
    put(body, im.flags, 8);               // flags: 64-bit, Linux host unless the entry says otherwise, compression
    // A compressed LTO-IR entry carries a word of the bitcode's own header here (measured).
    uint32_t lto_word = 0;
    if (compressed && im.kind == kFatbinLtoIr && !im.uncompressed_head.empty())
      std::memcpy(&lto_word, im.uncompressed_head.data(), std::min<size_t>(4, im.uncompressed_head.size()));
    put(body, lto_word, 8);
    put(body, compressed ? im.uncompressed_size : 0, 8);   // uncompressed size
    body += tail;
    body += payload;
  }
  std::string out;
  put(out, kFatbinMagic, 4);
  put(out, 1, 2);                         // container version
  put(out, sizeof(ContainerHeader), 2);
  put(out, body.size(), 8);
  out += body;
  return out;
}

namespace {

template <class T>
T read_le(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

bool is_elf(const uint8_t* p, size_t n) {
  return n >= 64 && p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F' && p[4] == 2;
}

constexpr uint16_t kEmCuda = 190;

}  // namespace

std::vector<std::pair<std::string, std::string>> elf_sections(const void* data, size_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  std::vector<std::pair<std::string, std::string>> out;
  if (!p || !is_elf(p, bytes)) return out;
  const uint64_t shoff = read_le<uint64_t>(p + 0x28);
  const uint16_t shentsize = read_le<uint16_t>(p + 0x3a);
  const uint16_t shnum = read_le<uint16_t>(p + 0x3c);
  const uint16_t shstrndx = read_le<uint16_t>(p + 0x3e);
  if (shnum == 0) return out;
  if (shentsize < 64 || shoff > bytes || uint64_t{shentsize} * shnum > bytes - shoff ||
      shstrndx >= shnum)
    throw Error::make(Err::InvalidValue, "ELF file's section table does not fit its ", bytes,
                      " bytes");
  auto section = [&](uint16_t i, uint64_t* off, uint64_t* size) {
    const uint8_t* sh = p + shoff + uint64_t{i} * shentsize;
    *off = read_le<uint64_t>(sh + 0x18);
    *size = read_le<uint64_t>(sh + 0x20);
    if (read_le<uint32_t>(sh + 4) == 8) *size = 0;   // SHT_NOBITS occupies no file bytes
    if (*off > bytes || *size > bytes - *off)
      throw Error::make(Err::InvalidValue, "ELF section ", i, " runs past the end of the file");
    return read_le<uint32_t>(sh);   // sh_name
  };
  uint64_t stroff = 0, strsize = 0;
  section(shstrndx, &stroff, &strsize);
  for (uint16_t i = 0; i < shnum; ++i) {
    uint64_t off = 0, size = 0;
    const uint32_t name_at = section(i, &off, &size);
    std::string name;
    if (name_at < strsize) {
      const char* n = reinterpret_cast<const char*>(p + stroff + name_at);
      name.assign(n, strnlen(n, static_cast<size_t>(strsize - name_at)));
    }
    out.emplace_back(std::move(name),
                     std::string(reinterpret_cast<const char*>(p + off), static_cast<size_t>(size)));
  }
  return out;
}

std::vector<std::string> host_object_fatbins(const void* data, size_t bytes, bool relocatable_only) {
  std::vector<std::string> out;
  for (const auto& [name, contents] : elf_sections(data, bytes)) {
    if (name != "__nv_relfatbin" && (relocatable_only || name != ".nv_fatbin")) continue;
    // A section holds containers back to back (a linked program has one per
    // translation unit), each starting on an 8-byte boundary.
    const uint8_t* s = reinterpret_cast<const uint8_t*>(contents.data());
    const size_t size = contents.size();
    size_t at = 0;
    while (at + sizeof(ContainerHeader) <= size) {
      if (read_le<uint32_t>(s + at) != kFatbinMagic) {
        at += 8;
        continue;
      }
      size_t used = 0;
      extract_images(s + at, size - at, &used, kFatbinPtx);
      out.emplace_back(reinterpret_cast<const char*>(s + at), used);
      at += (used + 7) / 8 * 8;
    }
  }
  return out;
}

std::vector<std::pair<std::string, std::string>> archive_members(const void* data, size_t bytes) {
  const char* p = static_cast<const char*>(data);
  std::vector<std::pair<std::string, std::string>> out;
  if (!p || bytes < 8 || std::memcmp(p, "!<arch>\n", 8) != 0)
    throw Error::make(Err::InvalidValue, "not an ar archive (no !<arch> signature)");
  std::string long_names;
  size_t at = 8;
  while (at + 60 <= bytes) {
    const char* h = p + at;
    if (h[58] != '`' || h[59] != '\n')
      throw Error::make(Err::InvalidValue, "ar archive member header at offset ", at,
                        " is malformed");
    const std::string size_field(h + 48, 10);
    char* end = nullptr;
    const unsigned long long size = std::strtoull(size_field.c_str(), &end, 10);
    if (end == size_field.c_str() || size > bytes - at - 60)
      throw Error::make(Err::InvalidValue, "ar archive member at offset ", at,
                        " runs past the end of the archive");
    std::string name(h, 16);
    while (!name.empty() && name.back() == ' ') name.pop_back();
    const char* body = h + 60;
    if (name == "//") {
      long_names.assign(body, size);   // GNU's table of names longer than 15 characters
    } else if (name == "/" || name == "/SYM64/") {
      // the symbol index: not a member
    } else {
      if (name.size() > 1 && name[0] == '/' && std::isdigit(static_cast<unsigned char>(name[1]))) {
        const size_t off = std::strtoul(name.c_str() + 1, nullptr, 10);
        if (off < long_names.size())
          name = long_names.substr(off, long_names.find('\n', off) - off);
      }
      if (!name.empty() && name.back() == '/') name.pop_back();
      out.emplace_back(name, std::string(body, size));
    }
    at += 60 + size + (size & 1);   // members start on even offsets
  }
  return out;
}

BlobKind classify_blob(const void* data, size_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  if (!p || bytes < 4) return BlobKind::Unknown;
  const uint32_t magic = read_le<uint32_t>(p);
  if (magic == kFatbinMagic || magic == kWrapperMagic) return BlobKind::Fatbin;
  if (is_elf(p, bytes))
    return read_le<uint16_t>(p + 18) == kEmCuda ? BlobKind::Cubin : BlobKind::HostObject;
  if (bytes >= 8 && std::memcmp(p, "!<arch>\n", 8) == 0) return BlobKind::Archive;
  // NVVM's LTO-IR (what nvcc -dlto and NVRTC's -dlto emit starts with
  // ed 43 4e 7f), and bare or wrapped LLVM bitcode.
  if (magic == 0x7f4e43edu || magic == 0xdec04342u || magic == 0x0b17c0deu) return BlobKind::LtoIr;
  // PTX: text whose first directive, past blank lines and comments, is .version.
  size_t i = 0;
  while (i < bytes) {
    const char c = static_cast<char>(p[i]);
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      ++i;
    } else if (c == '/' && i + 1 < bytes && p[i + 1] == '/') {
      while (i < bytes && p[i] != '\n') ++i;
    } else if (c == '/' && i + 1 < bytes && p[i + 1] == '*') {
      i += 2;
      while (i + 1 < bytes && !(p[i] == '*' && p[i + 1] == '/')) ++i;
      i += 2;
    } else {
      break;
    }
  }
  if (bytes - std::min(i, bytes) >= 8 && std::memcmp(p + i, ".version", 8) == 0) return BlobKind::Ptx;
  return BlobKind::Unknown;
}

bool cubin_linked(const void* data, size_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  return p && is_elf(p, bytes) && read_le<uint16_t>(p + 16) == 2;   // ET_EXEC
}

uint32_t cubin_arch(const void* data, size_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  if (!p || !is_elf(p, bytes) || read_le<uint16_t>(p + 18) != kEmCuda) return 0;
  // e_flags names the SM. Its place moved with the CUDA ELF ABI version
  // (e_ident[8]), as cubins from the two toolkits show: CUDA 12.0's (ABI 7)
  // carry sm_86 as 0x560556, the low byte; CUDA 13.0's (ABI 8) as 0x6005604,
  // the second byte -- 0x5a04 for sm_90a, 0x7802 for sm_120.
  const uint32_t flags = read_le<uint32_t>(p + 0x30);
  return p[8] >= 8 ? (flags >> 8) & 0xff : flags & 0xff;
}


}  // namespace vgpu::cuda
