// A tile of doubles for the matrix path, positioned in the cache.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vgpu::amd {

// One of the matrix path's tiles (the operands as doubles and the result), a std::vector<double> in all but where it starts: the
// start is moved so that its address is K * 1024 bytes into a 4 KB page. The four tiles are stored to and loaded from in the
// same loop; allocated one after the other from the heap they sat wherever the heap's history put them, and where a store's and a later
// load's addresses agree in the low twelve bits the load waits for the store (4K aliasing), which made v_mfma_f64 23% slower in one
// layout of the heap than in another. Values are those of the vector: a prefix survives a resize, new elements are zero.
template <int K>
class TileBuf {
 public:
  double* data() { return p_; }
  const double* data() const { return p_; }
  size_t size() const { return n_; }
  double* begin() { return data(); }
  double* end() { return data() + n_; }
  const double* begin() const { return data(); }
  const double* end() const { return data() + n_; }
  double& operator[](size_t i) { return data()[i]; }
  const double& operator[](size_t i) const { return data()[i]; }
  void resize(size_t m) {
    if (m + kPad > store_.size()) {
      std::vector<double> ns(m + kPad);
      const size_t noff = offset_in(ns.data());
      if (n_) std::memcpy(ns.data() + noff, data(), std::min(n_, m) * sizeof(double));
      store_.swap(ns);
      off_ = noff;
      p_ = store_.data() + off_;
    } else if (m > n_) {
      std::fill(data() + n_, data() + m, 0.0);
    }
    n_ = m;
  }

 private:
  static constexpr size_t kPad = 512;   // doubles: a page
  static size_t offset_in(const double* base) {
    const uintptr_t at = reinterpret_cast<uintptr_t>(base);
    const uintptr_t want = static_cast<uintptr_t>(K) * 1024;
    return static_cast<size_t>(((want + 4096 - (at & 4095)) & 4095) / sizeof(double));
  }
  std::vector<double> store_;
  size_t off_ = 0, n_ = 0;
  double* p_ = nullptr;
};

}  // namespace vgpu::amd
