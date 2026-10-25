// TileBuf (amd/src/tile_buf.hpp), the matrix path's tile: a std::vector<double> in everything but where it starts. The prefix survives a
// resize and the new elements are zero (as a vector's are, also when the tile shrank and grew again), and its address is K * 1024 bytes
// into a page, wherever the heap puts the storage.
#include <cstdint>
#include <vector>

#include "../../src/tile_buf.hpp"
#include "vtest.hpp"

using vgpu::amd::TileBuf;

template <int K>
static void check_offset(TileBuf<K>& t) {
  VCHECK_EQ(reinterpret_cast<uintptr_t>(t.data()) % 4096, uintptr_t{K} * 1024);
}

VTEST(a_tile_starts_where_it_is_told_in_the_page) {
  TileBuf<0> a;
  TileBuf<2> b;
  TileBuf<3> c;
  for (size_t n : {size_t{1}, size_t{64}, size_t{300}, size_t{5000}, size_t{20}}) {
    std::vector<double> noise(n * 3 + 7);   // moves the heap about
    a.resize(n), b.resize(n), c.resize(n);
    check_offset(a), check_offset(b), check_offset(c);
    VCHECK_EQ(a.size(), n);
    VCHECK_EQ(static_cast<size_t>(a.end() - a.begin()), n);
  }
}

VTEST(a_tile_resizes_like_a_vector) {
  TileBuf<1> t;
  std::vector<double> v;
  const size_t sizes[] = {8, 40, 4, 600, 3, 500, 3, 1500, 9, 2000, 100};
  for (size_t n : sizes) {
    t.resize(n), v.resize(n);
    for (size_t i = 0; i < n; ++i) VCHECK_EQ(t[i], v[i]);
    for (size_t i = 0; i < n; ++i) t[i] = v[i] = static_cast<double>(n * 1000 + i) / 7;
    // (only a prefix that is still there matters: stale elements past a shrink are zeroed again by the next growth)
    check_offset(t);
  }
}

VTEST_MAIN
