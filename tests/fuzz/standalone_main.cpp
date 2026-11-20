// A main() for the fuzz targets when they are not built with libFuzzer (-DVGPU_FUZZ=OFF, the default): it runs
// LLVMFuzzerTestOneInput on every file named on the command line, and on every file under a directory named there.
// That is how the seed corpus and the inputs of past crashes (tests/fuzz/regress) run as ordinary ctest tests.
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);
extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) __attribute__((weak));

static size_t run_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  // A copy of exactly the right size, so a read past the end is the sanitizer's to catch.
  uint8_t* exact = new uint8_t[bytes.empty() ? 1 : bytes.size()];
  std::copy(bytes.begin(), bytes.end(), exact);
  LLVMFuzzerTestOneInput(exact, bytes.size());
  delete[] exact;
  return 1;
}

int main(int argc, char** argv) {
  if (LLVMFuzzerInitialize) LLVMFuzzerInitialize(&argc, &argv);
  size_t n = 0;
  for (int i = 1; i < argc; ++i) {
    const std::filesystem::path p(argv[i]);
    if (std::filesystem::is_directory(p)) {
      std::vector<std::filesystem::path> files;
      for (const auto& e : std::filesystem::recursive_directory_iterator(p))
        if (e.is_regular_file()) files.push_back(e.path());
      std::sort(files.begin(), files.end());
      for (const auto& f : files) n += run_file(f);
    } else if (std::filesystem::exists(p)) {
      n += run_file(p);
    } else {
      std::fprintf(stderr, "no such input: %s\n", argv[i]);
      return 2;
    }
  }
  std::printf("ran %zu inputs\n", n);
  return n ? 0 : 2;   // nothing to run is a mistake: the seed corpus did not generate
}
