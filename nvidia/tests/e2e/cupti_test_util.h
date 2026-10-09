// Shared by the CUPTI cases that print what a trace holds so it can be compared
// line by line with what NVIDIA's libcupti printed for the same program
// (nvidia/tests/e2e/run_cupti_case.sh): a buffer for records, a printer that
// does not depend on time or on the numbers a CUPTI picks for its ids, and the
// ranks that stand in for those ids.
//
// Correlation, stream, context, module and graph ids differ between CUPTI
// implementations and mean nothing across them; what means something is which
// records share one. A line carries an id as "#c<raw>", "#s<raw>" and so on
// until the end, when each id becomes its rank among those the run saw, in the
// order they were first printed: "c0" is the first correlation id.
#ifndef VGPU_CUPTI_TEST_UTIL_H
#define VGPU_CUPTI_TEST_UTIL_H

#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cupti.h>

namespace cupti_test {

inline std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char* f, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

// Output, in the order it happened. Ids are rewritten when it is printed.
inline std::vector<std::string>& lines() {
  static std::vector<std::string> v;
  return v;
}
inline void out(const std::string& s) { lines().push_back(s); }

// Replaces every "#<tag><digits>" with "<tag><rank>", ranks by first appearance.
inline void print_all() {
  const char tags[] = {'c', 's', 'x', 'g', 'n', 'm', 'e'};
  std::map<char, std::map<uint64_t, int>> rank;
  for (const std::string& l : lines())
    for (char t : tags)
      for (size_t at = 0; (at = l.find(std::string("#") + t, at)) != std::string::npos;) {
        at += 2;
        const uint64_t raw = std::strtoull(l.c_str() + at, nullptr, 10);
        auto& r = rank[t];
        if (!r.count(raw)) r[raw] = static_cast<int>(r.size());
      }
  for (std::string l : lines()) {
    for (char t : tags) {
      const std::string key = std::string("#") + t;
      for (size_t at = 0; (at = l.find(key, at)) != std::string::npos;) {
        size_t end = at + 2;
        while (end < l.size() && std::isdigit(static_cast<unsigned char>(l[end]))) ++end;
        const uint64_t raw = std::strtoull(l.c_str() + at + 2, nullptr, 10);
        const std::string rep = std::string(1, t) + std::to_string(rank[t].at(raw));
        l.replace(at, end - at, rep);
        at += rep.size();
      }
    }
    std::printf("%s\n", l.c_str());
  }
}

inline const char* mem_kind(uint32_t k) {
  switch (k) {
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE: return "pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED: return "pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE: return "device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY: return "array";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED: return "managed";
    default: return "unknown";
  }
}

// A buffer for records that the consumer owns, as CUPTI asks.
inline void CUPTIAPI request_buffer(uint8_t** buffer, size_t* size, size_t* max_records) {
  static uint8_t storage[1 << 20] __attribute__((aligned(8)));
  std::memset(storage, 0, sizeof storage);
  *buffer = storage;
  *size = sizeof storage;
  *max_records = 0;
}

}  // namespace cupti_test

#endif  // VGPU_CUPTI_TEST_UTIL_H
