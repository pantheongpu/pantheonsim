/* Bring-up software's view of the register model: a plain C program against
 * vgpu_regs.h and libvgpuregs only. It sizes a BAR, talks to an AMD GPU's SMU
 * through its mailbox, polls engine status, and checks the refusals. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vgpu_regs.h"

static int failures = 0;
#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #cond,           \
              vgpu_regs_last_error());                                              \
      ++failures;                                                                   \
    }                                                                               \
  } while (0)

static uint32_t reg(const char* space, const char* name) {
  uint32_t off = 0, width = 0;
  CHECK(vgpu_regs_find(space, name, &off, &width) == VGPU_REGS_OK);
  return off;
}

int main(void) {
  vgpu_regs* cfg = NULL;
  vgpu_regs* mmio = NULL;
  uint32_t v = 0;

  /* Configuration space: identity, and a BAR sizing probe. */
  CHECK(vgpu_regs_open(0, "config", &cfg) == VGPU_REGS_OK);
  CHECK(vgpu_regs_read(cfg, 0x00, 2, &v) == VGPU_REGS_OK && v == 0x1002);
  CHECK(vgpu_regs_write(cfg, reg("config", "bar5"), 4, 0xffffffffu) == VGPU_REGS_OK);
  CHECK(vgpu_regs_read(cfg, reg("config", "bar5"), 4, &v) == VGPU_REGS_OK);
  CHECK((~(v & ~0xfu) + 1) == (512u << 10));   /* 512 KiB of registers */
  CHECK(vgpu_regs_read(cfg, 0x01, 2, &v) == VGPU_REGS_EINVAL);
  CHECK(strlen(vgpu_regs_last_error()) > 0);

  /* MMIO: the SMU mailbox, as a driver drives it. */
  CHECK(vgpu_regs_open(0, "mmio", &mmio) == VGPU_REGS_OK);
  const uint32_t msg = reg("mmio", "smu_message"), arg = reg("mmio", "smu_argument"),
                 resp = reg("mmio", "smu_response");
  CHECK(vgpu_regs_write(mmio, resp, 4, 0) == VGPU_REGS_OK);
  CHECK(vgpu_regs_write(mmio, arg, 4, 99) == VGPU_REGS_OK);
  CHECK(vgpu_regs_write(mmio, msg, 4, 0x1) == VGPU_REGS_OK); /* TestMessage */
  CHECK(vgpu_regs_read(mmio, resp, 4, &v) == VGPU_REGS_OK && v == 0x1);
  CHECK(vgpu_regs_read(mmio, arg, 4, &v) == VGPU_REGS_OK && v == 100);
  CHECK(vgpu_regs_read(mmio, reg("mmio", "grbm_status"), 4, &v) == VGPU_REGS_OK && (v >> 31) == 0);
  CHECK(vgpu_regs_read(mmio, 0x8010, 2, &v) == VGPU_REGS_EINVAL);

  /* Refusals. */
  vgpu_regs* none = NULL;
  CHECK(vgpu_regs_open(7, "config", &none) == VGPU_REGS_ENODEV && none == NULL);
  CHECK(vgpu_regs_open(0, "bogus", &none) == VGPU_REGS_EINVAL);
  CHECK(vgpu_regs_find("mmio", "no_such_register", &v, NULL) == VGPU_REGS_ENOENT);
  CHECK(vgpu_regs_read(NULL, 0, 4, &v) == VGPU_REGS_EINVAL);

  vgpu_regs_close(cfg);
  vgpu_regs_close(mmio);
  if (!failures) printf("regs C API: all checks passed\n");
  return failures ? 1 : 0;
}
