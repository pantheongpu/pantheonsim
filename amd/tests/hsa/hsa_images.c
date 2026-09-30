/* HSA's images extension, as a program written against ROCm's runtime uses
 * it: the agent's image limits, an image made over memory the program
 * allocated, texels imported, exported and cleared, a linear layout with a
 * pitch of its own, a 1D buffer image, and samplers. Built against
 * VirtualGPU's HSA header or ROCm's (amd/tests/e2e/run_hsa.sh).
 *
 *   hsa_images
 *
 * On a GPU with texture units (a Radeon profile) every check runs; on one
 * without (MI300X) the limits are 0 and an image is refused. One line a
 * check, "ok <what>" or "FAIL <what>". */
#ifdef VGPU_REAL_HSA
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_image.h>
#else
#include "vgpu/hsa_abi.h"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static void check(const char* what, int ok) {
  printf("%s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define MUST(x)                                                                  \
  do {                                                                           \
    hsa_status_t s_ = (x);                                                       \
    if (s_ != HSA_STATUS_SUCCESS) {                                              \
      const char* why_ = "?";                                                    \
      hsa_status_string(s_, &why_);                                              \
      printf("FAIL %s: %s\n", #x, why_);                                         \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

/* The extension's numbers, spelled out so either header builds this. */
enum { GEOMETRY_1D = 0, GEOMETRY_2D = 1, GEOMETRY_1DB = 5 };
enum { TYPE_UNORM_INT8 = 2, TYPE_FLOAT = 15, ORDER_R = 1, ORDER_RGBA = 8, ORDER_BGRA = 9 };

static hsa_agent_t gpu, cpu;
static int found;
static hsa_status_t first_gpu(hsa_agent_t a, void* data) {
  (void)data;
  hsa_device_type_t type;
  MUST(hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type));
  if (type == HSA_DEVICE_TYPE_GPU && !found) gpu = a, found = 1;
  if (type == HSA_DEVICE_TYPE_CPU) cpu = a;
  return HSA_STATUS_SUCCESS;
}
static hsa_amd_memory_pool_t system_pool;
static int have_pool;
static hsa_status_t each_pool(hsa_amd_memory_pool_t p, void* data) {
  (void)data;
  hsa_amd_segment_t segment;
  MUST(hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));
  if (segment == HSA_AMD_SEGMENT_GLOBAL && !have_pool) system_pool = p, have_pool = 1;
  return HSA_STATUS_SUCCESS;
}

int main(void) {
  MUST(hsa_init());
  MUST(hsa_iterate_agents(first_gpu, NULL));
  uint8_t extensions[128];
  MUST(hsa_agent_get_info(gpu, HSA_AGENT_INFO_EXTENSIONS, extensions));
  check("the GPU agent has the images extension", (extensions[0] >> HSA_EXTENSION_IMAGES) & 1);
  uint32_t dims2[2] = {0, 0}, rorw = 0, samplers = 0;
  MUST(hsa_agent_get_info(gpu, (hsa_agent_info_t)HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS, dims2));
  MUST(hsa_agent_get_info(gpu, (hsa_agent_info_t)HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES, &rorw));
  MUST(hsa_agent_get_info(gpu, (hsa_agent_info_t)HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS, &samplers));

  hsa_ext_image_descriptor_t d;
  memset(&d, 0, sizeof d);
  d.geometry = GEOMETRY_2D;
  d.width = 8;
  d.height = 4;
  d.format.channel_type = TYPE_UNORM_INT8;
  d.format.channel_order = ORDER_RGBA;
  if (rorw == 0) {
    check("an agent without texture units has no image limits", dims2[0] == 0 && dims2[1] == 0 && samplers == 0);
    hsa_ext_image_data_info_t info;
    check("and refuses an image's format",
          hsa_ext_image_data_get_info(gpu, &d, HSA_ACCESS_PERMISSION_RW, &info) ==
              (hsa_status_t)HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED);
    MUST(hsa_shut_down());
    return failures ? 1 : 0;
  }
  check("a Radeon agent's image limits are ROCm's", dims2[0] == 16384 && dims2[1] == 16384 && rorw == 64 &&
                                                        samplers == 16);
  /* System memory: the host reads it, and every GPU reaches it. */
  MUST(hsa_amd_agent_iterate_memory_pools(cpu, each_pool, NULL));

  /* An RGBA8 image over memory of the program's. */
  hsa_ext_image_data_info_t info;
  MUST(hsa_ext_image_data_get_info(gpu, &d, HSA_ACCESS_PERMISSION_RW, &info));
  check("an 8x4 RGBA8 image takes 128 bytes, aligned to 256", info.size == 128 && info.alignment == 256);
  void* texels = NULL;
  MUST(hsa_amd_memory_pool_allocate(system_pool, 4096, 0, &texels));
  hsa_ext_image_t image;
  MUST(hsa_ext_image_create(gpu, &d, texels, HSA_ACCESS_PERMISSION_RW, &image));
  const uint32_t* words = (const uint32_t*)(uintptr_t)image.handle;
  const uint32_t height = ((words[2] >> 14) & 0x3FFF) + 1;
  check("its resource holds the height where HIP's texture functions read it, and the width, type and order after it",
        (height == 4 || ((words[2] >> 14) & 0xFFFF) + 1 == 4) && words[8] == TYPE_UNORM_INT8 &&
            words[9] == ORDER_RGBA && words[10] == 8);
  uint8_t in[128], out[128];
  for (int i = 0; i < 128; ++i) in[i] = (uint8_t)(i * 7 + 3);
  hsa_ext_image_region_t all = {{0, 0, 0}, {8, 4, 1}};
  MUST(hsa_ext_image_import(gpu, in, 32, 0, image, &all));
  memset(out, 0, sizeof out);
  MUST(hsa_ext_image_export(gpu, image, out, 32, 0, &all));
  check("texels imported come back exported", memcmp(in, out, 128) == 0 && memcmp(texels, in, 128) == 0);
  /* Clearing texel (1..2, 1): red and alpha 1.0, green and blue 0. */
  const float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  hsa_ext_image_region_t two = {{1, 1, 0}, {2, 1, 1}};
  MUST(hsa_ext_image_clear(gpu, image, red, &two));
  MUST(hsa_ext_image_export(gpu, image, out, 32, 0, &all));
  const uint8_t* t = out + 32 + 4;
  check("a clear writes the value in the image's format, and only the region",
        t[0] == 255 && t[1] == 0 && t[2] == 0 && t[3] == 255 && t[4] == 255 && t[7] == 255 &&
            memcmp(out, in, 36) == 0 && memcmp(out + 44, in + 44, 128 - 44) == 0);
  MUST(hsa_ext_image_destroy(gpu, image));

  /* A linear layout whose rows are 256 bytes apart. */
  hsa_ext_image_t pitched;
  d.format.channel_type = TYPE_FLOAT;
  d.format.channel_order = ORDER_R;
  MUST(hsa_ext_image_create_with_layout(gpu, &d, texels, HSA_ACCESS_PERMISSION_RW,
                                        HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR, 256, 0, &pitched));
  float row[8], back[32];
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 8; ++x) row[x] = (float)(x + 10 * y);
    hsa_ext_image_region_t one = {{0, (uint32_t)y, 0}, {8, 1, 1}};
    MUST(hsa_ext_image_import(gpu, row, 0, 0, pitched, &one));
  }
  MUST(hsa_ext_image_export(gpu, pitched, back, 0, 0, &all));
  int wrong = 0;
  for (int i = 0; i < 32; ++i) wrong += back[i] != (float)(i % 8 + 10 * (i / 8));
  wrong += ((const float*)texels)[64 + 3] != 13.0f;   /* row 1 starts 256 bytes in */
  check("a pitched linear image keeps its rows the pitch apart", wrong == 0);
  MUST(hsa_ext_image_destroy(gpu, pitched));

  /* A 1D buffer image: a buffer resource, a texel to a record. */
  hsa_ext_image_descriptor_t b;
  memset(&b, 0, sizeof b);
  b.geometry = GEOMETRY_1DB;
  b.width = 100;
  b.format.channel_type = TYPE_UNORM_INT8;
  b.format.channel_order = ORDER_BGRA;
  hsa_ext_image_t buffer;
  MUST(hsa_ext_image_create(gpu, &b, texels, HSA_ACCESS_PERMISSION_RO, &buffer));
  const uint32_t* v = (const uint32_t*)(uintptr_t)buffer.handle;
  check("a 1D buffer image is a buffer resource: its base, a 4-byte stride, 100 records, BGRA's swizzle",
        v[0] == (uint32_t)(uintptr_t)texels && ((v[1] >> 16) & 0x3FFF) == 4 && v[2] == 100 && (v[3] & 7) == 6 &&
            ((v[3] >> 6) & 7) == 4);
  MUST(hsa_ext_image_destroy(gpu, buffer));

  /* Samplers: unnormalized and point, normalized and bilinear. */
  hsa_ext_sampler_descriptor_v2_t sd;
  memset(&sd, 0, sizeof sd);
  sd.coordinate_mode = 0;
  sd.filter_mode = 0;
  sd.address_modes[0] = sd.address_modes[1] = sd.address_modes[2] = 1;
  hsa_ext_sampler_t s1, s2;
  MUST(hsa_ext_sampler_create_v2(gpu, &sd, &s1));
  sd.coordinate_mode = 1;
  sd.filter_mode = 1;
  sd.address_modes[0] = 3;
  MUST(hsa_ext_sampler_create_v2(gpu, &sd, &s2));
  const uint32_t* w1 = (const uint32_t*)(uintptr_t)s1.handle;
  const uint32_t* w2 = (const uint32_t*)(uintptr_t)s2.handle;
  check("a sampler's resource holds the bits HIP's texture functions read",
        ((w1[0] >> 15) & 1) == 1 && ((w1[2] >> 20) & 3) == 0 && ((w2[0] >> 15) & 1) == 0 &&
            ((w2[2] >> 20) & 3) == 1 && (w2[0] & 7) == 0 && ((w1[0]) & 7) == 2);
  MUST(hsa_ext_sampler_destroy(gpu, s1));
  MUST(hsa_ext_sampler_destroy(gpu, s2));
  check("an image already destroyed is refused", hsa_ext_image_destroy(gpu, buffer) != HSA_STATUS_SUCCESS);
  MUST(hsa_amd_memory_pool_free(texels));
  MUST(hsa_shut_down());
  return failures ? 1 : 0;
}
