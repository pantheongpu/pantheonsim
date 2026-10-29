// A file of the cubins sass::unsupported has found it runs in full, kept between processes. See ok_cache.cpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace vgpu::sass::okcache {

struct Key {
  uint64_t a = 0, b = 0;
};

// A 128-bit hash of the image's every byte (and its length).
Key key_of(const uint8_t* image, size_t size);

// True when an earlier process (or this one) recorded that the image with this key ran in full, by this very build of
// the simulator. False when there is no file, the file cannot be trusted, or the cache is off (it is opt-in: VGPU_SASS_CACHE=1).
bool known(const Key& k);

// Records that the image with this key ran in full. Does nothing when the cache is off, unwritable, or full.
void remember(const Key& k);

// How many known() calls answered true (for tests).
uint64_t hits();

}  // namespace vgpu::sass::okcache
