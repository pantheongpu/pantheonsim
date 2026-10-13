// nvJPEG's host decoder on damaged files, against the digests of what it decoded before its entropy decoder was reworked for
// speed: bits flipped in the entropy-coded data, a file cut short, markers and stuffing bytes planted in it, a byte of the
// tables changed, junk for an end marker. The decoder reads zeros past the end of a scan and after a marker, finds a restart
// marker from the byte after the last bit it read, and gives up on a code at 17 bits; each of those is part of what these
// digests pin down. (A simulator-only check: NVIDIA's decoder answers damaged files its own way, and the files are made here.)
//
//   nvjpeg_damaged [fixture directory [--print]]
//
// --print writes the digest table (nvjpeg_damaged_expected.inc).
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static std::string dir = "nvidia/tests/data/jpeg";
static std::vector<unsigned char> slurp(const std::string& name) {
  std::ifstream f(dir + "/" + name, std::ios::binary);
  return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// splitmix64: the same numbers on every platform.
struct Rng {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  size_t below(size_t n) { return static_cast<size_t>(next() % n); }
};

static uint64_t fnv(const unsigned char* p, size_t n, uint64_t x = 1469598103934665603ull) {
  for (size_t i = 0; i < n; ++i) {
    x ^= p[i];
    x *= 1099511628211ull;
  }
  return x;
}

static const char* kFixtures[] = {"b420.jpg", "b422.jpg", "b444.jpg", "big_pr.jpg", "big_q5.jpg", "cmyk2.jpg",
                                  "gray.jpg", "p420.jpg", "p444.jpg", "pgray.jpg", "r420.jpg"};
constexpr int kVariants = 18;

// The damaged file of fixture `f`, variant `k` (0 .. kVariants - 1).
static std::vector<unsigned char> damaged(const std::vector<unsigned char>& d, Rng& rng, int k) {
  std::vector<unsigned char> v = d;
  size_t sos = 0;
  for (size_t i = 0; i + 1 < d.size(); ++i)
    if (d[i] == 0xFF && d[i + 1] == 0xDA) {
      sos = i;
      break;
    }
  const size_t start = sos ? sos + 12 : 100;
  if (start + 8 >= v.size()) return v;
  auto in_data = [&] { return start + rng.below(v.size() - start); };
  switch (k % 6) {
    case 0:
      for (size_t n = 1 + rng.below(6); n; --n) v[in_data()] ^= static_cast<unsigned char>(1u << rng.below(8));
      break;
    case 1: v.resize(in_data()); break;
    case 2:
      for (int n = 0; n < 40; ++n) v[in_data()] ^= static_cast<unsigned char>(1u << rng.below(8));
      break;
    case 3:
      for (int n = 0; n < 8; ++n) {
        static const unsigned char plant[] = {0xFF, 0x00, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7};
        v[in_data()] = plant[rng.below(sizeof plant)];
      }
      break;
    case 4: v[2 + rng.below(start - 4 > 3 ? start - 6 : 1)] = static_cast<unsigned char>(rng.below(256)); break;
    default:
      v.resize(v.size() - 2);
      for (int n = 0; n < 5; ++n) v.push_back(static_cast<unsigned char>(rng.below(256)));
      break;
  }
  return v;
}

int main(int argc, char** argv) {
  bool print = false;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--print")) print = true;
    else dir = argv[a];
  }
  if (const char* e = std::getenv("VGPU_E2E_DATA"); e && *e && argc < 2) dir = std::string(e) + "/jpeg";
  // The digests are this simulator's; NVIDIA's decoder answers damaged files its own way.
  if (!dlsym(RTLD_DEFAULT, "_Z22vgpu_device_allocationPKvPPvPm")) {
    std::printf("SKIP: simulator only\n");
    return 0;
  }
  nvjpegHandle_t h;
  if (nvjpegCreateSimple(&h)) return 1;
  nvjpegJpegState_t st = nullptr;
  nvjpegJpegStateCreate(h, &st);
  const size_t cap = 1 << 20;
  unsigned char* dev = nullptr;
  cudaMalloc(&dev, cap * 4);
  // Released on every way out of main (a leak check, ASan's among them, covers the test too).
  struct Release {
    nvjpegHandle_t h;
    nvjpegJpegState_t st;
    unsigned char* dev;
    ~Release() {
      if (dev) cudaFree(dev);
      if (st) nvjpegJpegStateDestroy(st);
      nvjpegDestroy(h);
    }
  } release{h, st, dev};
  std::vector<unsigned char> host(cap * 4);
  std::vector<uint64_t> digests;
  Rng rng{20261012};
  for (const char* name : kFixtures) {
    const std::vector<unsigned char> src = slurp(name);
    if (src.empty()) {
      std::printf("FAIL: cannot read %s/%s\n", dir.c_str(), name);
      return 1;
    }
    for (int k = 0; k < kVariants; ++k) {
      const std::vector<unsigned char> d = damaged(src, rng, k);
      uint64_t digest = 1469598103934665603ull;
      for (const nvjpegOutputFormat_t fmt : {NVJPEG_OUTPUT_UNCHANGED, NVJPEG_OUTPUT_RGBI}) {
        int nc;
        nvjpegChromaSubsampling_t ss;
        int wd[NVJPEG_MAX_COMPONENT], ht[NVJPEG_MAX_COMPONENT];
        const int info = nvjpegGetImageInfo(h, d.data(), d.size(), &nc, &ss, wd, ht);
        digest = fnv(reinterpret_cast<const unsigned char*>(&info), sizeof info, digest);
        if (info) continue;
        nvjpegImage_t img;
        std::memset(&img, 0, sizeof img);
        const size_t plane = static_cast<size_t>(wd[0]) * ht[0] + 64;
        if (plane * 4 > cap) continue;
        for (int i = 0; i < 4; ++i) {
          img.channel[i] = dev + static_cast<size_t>(i) * plane;
          img.pitch[i] = static_cast<size_t>(wd[0]);
        }
        if (fmt == NVJPEG_OUTPUT_RGBI) img.pitch[0] = static_cast<size_t>(wd[0]) * 3;
        if (fmt == NVJPEG_OUTPUT_UNCHANGED)
          for (int i = 0; i < nc && i < 4; ++i) img.pitch[i] = static_cast<size_t>(wd[i] > 0 ? wd[i] : wd[0]);
        cudaMemset(dev, 0xAB, plane * 4);
        const int rc = nvjpegDecode(h, st, d.data(), d.size(), fmt, &img, 0);
        cudaDeviceSynchronize();
        digest = fnv(reinterpret_cast<const unsigned char*>(&rc), sizeof rc, digest);
        if (rc == 0) {
          cudaMemcpy(host.data(), dev, plane * 4, cudaMemcpyDeviceToHost);
          digest = fnv(host.data(), plane * 4, digest);
        }
      }
      digests.push_back(digest);
    }
  }
  if (print) {
    for (size_t i = 0; i < digests.size(); ++i) std::printf("0x%016llxull,%s", static_cast<unsigned long long>(digests[i]), i % 3 == 2 ? "\n" : " ");
    std::printf("\n");
    return 0;
  }
  static const uint64_t kExpected[] = {
#include "nvjpeg_damaged_expected.inc"
  };
  int bad = 0;
  if (digests.size() != sizeof kExpected / sizeof kExpected[0]) {
    std::printf("FAIL: %zu cases, %zu expected\n", digests.size(), sizeof kExpected / sizeof kExpected[0]);
    return 1;
  }
  for (size_t i = 0; i < digests.size(); ++i)
    if (digests[i] != kExpected[i]) {
      std::printf("FAIL: %s variant %zu: digest %016llx, expected %016llx\n", kFixtures[i / kVariants], i % kVariants,
                  static_cast<unsigned long long>(digests[i]), static_cast<unsigned long long>(kExpected[i]));
      ++bad;
    }
  if (bad) return 1;
  std::printf("PASS: %zu damaged files decode as they did\n", digests.size());
  return 0;
}
