/* VirtualGPU register access for bring-up software: a C interface to a
 * simulated GPU's register spaces -- the PCI configuration space of any GPU,
 * and an AMD GPU's MMIO registers behind BAR5 -- through the same register
 * model as `vgpu regs` (docs/registers.md). Reads reflect the device as it is
 * at the moment of the read; writes follow each register's semantics and are
 * seen by every process on the machine; every access is logged.
 *
 * Link with -lvgpuregs. The machine is the one `vgpu smi` describes: a running
 * VirtualGPU, or VGPU_GPU and VGPU_DEVICE_COUNT. */
#ifndef VGPU_REGS_H
#define VGPU_REGS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vgpu_regs vgpu_regs;

enum {
  VGPU_REGS_OK = 0,
  VGPU_REGS_EINVAL = -1,   /* a bad argument: a misaligned or out-of-range access, an unknown space */
  VGPU_REGS_ENODEV = -2,   /* no such GPU, or no machine to describe */
  VGPU_REGS_ENOTSUP = -3,  /* the GPU has no such space modelled (MMIO on an NVIDIA GPU) */
  VGPU_REGS_ENOENT = -4,   /* no register by that name */
  VGPU_REGS_ERROR = -5     /* anything else; vgpu_regs_last_error says what */
};

/* Opens `space` ("config" or "mmio") of GPU `gpu`. */
int vgpu_regs_open(int gpu, const char* space, vgpu_regs** out);
void vgpu_regs_close(vgpu_regs* regs);

/* An access of `size` bytes: 1, 2 or 4 in configuration space, naturally
 * aligned; 4 in MMIO, aligned. */
int vgpu_regs_read(vgpu_regs* regs, uint32_t offset, uint32_t size, uint32_t* value);
int vgpu_regs_write(vgpu_regs* regs, uint32_t offset, uint32_t size, uint32_t value);

/* A register of `space` by the name the database gives it: its offset, and its
 * width in bits. */
int vgpu_regs_find(const char* space, const char* name, uint32_t* offset, uint32_t* width_bits);

/* What went wrong with this thread's last call, or "" after a success. */
const char* vgpu_regs_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* VGPU_REGS_H */
