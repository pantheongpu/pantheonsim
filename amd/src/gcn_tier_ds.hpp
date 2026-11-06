// The wide (two to four words a lane) LDS accesses of the AMD tier's ds_read / ds_write handlers (gcn_tier.cpp h_ds), in a translation unit of
// their own that is compiled without link-time optimization: with them in the tier's unit, their size moved the inlining and the layout of the
// unrelated memory handlers (flat and global accesses read 2 to 12% slower with equal instruction counts). Bits only move; no arithmetic.
#pragma once

#include <cstdint>

namespace vgpu::amd::gcn::tier_ds {

// A run of cnt consecutive lanes: `rows` are the register rows already offset to the first lane, `at` the run's first byte of LDS, w words a lane (2 or 4).
void ds_run_read(uint32_t* const* rows, const uint8_t* at, unsigned cnt, unsigned w);
void ds_run_write(uint32_t* const* rows, uint8_t* at, unsigned cnt, unsigned w);
// All n lanes (a multiple of 4) at the addresses addr[l] + x0, w words a lane (2 or 4), stored in lane order.
void ds_all_read(uint32_t* const* rows, const uint8_t* lds, const uint32_t* addr, uint32_t x0, unsigned w, unsigned n);
void ds_all_write(uint32_t* const* rows, uint8_t* lds, const uint32_t* addr, uint32_t x0, unsigned w, unsigned n);
// The lanes of `ex` one by one, w words (2 to 4) a lane.
void ds_lanes_read(uint32_t* const* rows, const uint8_t* lds, const uint32_t* addr, uint32_t x0, uint64_t ex, unsigned w);
void ds_lanes_write(uint32_t* const* rows, uint8_t* lds, const uint32_t* addr, uint32_t x0, uint64_t ex, unsigned w);
// ds_read2_b64 and ds_write2_b64 (and st64): `rows` are the four registers of the two 64-bit values, x0 and x1 the byte offsets; 0 when a lane's
// access is not inside the LDS (nothing was changed), else 1.
int ds_two_read(uint32_t* const* rows, const uint32_t* addr_row, uint32_t x0, uint32_t x1, uint64_t ex, uint8_t* lds, uint64_t size, unsigned n);
int ds_two_write(const uint32_t* const* rows, const uint32_t* addr_row, uint32_t x0, uint32_t x1, uint64_t ex, uint8_t* lds, uint64_t size, unsigned n);

}  // namespace vgpu::amd::gcn::tier_ds
