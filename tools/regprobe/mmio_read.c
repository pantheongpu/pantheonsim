/* Reads a GPU's registers through a PCI BAR, as root, for tools/regprobe.
 *
 *   mmio_read RESOURCE_FILE START END [STRIDE]
 *
 * Maps the BAR (/sys/bus/pci/devices/<bdf>/resourceN) read-only and prints
 * "offset value" for every 32-bit register from START up to END, each line
 * flushed as it is read: if a read hangs the device, everything read before it
 * is already on disk. It never writes a register. */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: mmio_read RESOURCE_FILE START END [STRIDE]\n");
    return 2;
  }
  const uint64_t start = strtoull(argv[2], NULL, 0), end = strtoull(argv[3], NULL, 0);
  const uint64_t stride = argc > 4 ? strtoull(argv[4], NULL, 0) : 4;
  if (start % 4 || stride % 4 || stride == 0 || end < start) {
    fprintf(stderr, "mmio_read: START and STRIDE are multiples of 4, END is not below START\n");
    return 2;
  }
  const int fd = open(argv[1], O_RDONLY | O_SYNC);
  if (fd < 0) {
    fprintf(stderr, "mmio_read: %s: %s\n", argv[1], strerror(errno));
    return 1;
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || (uint64_t)st.st_size < end) {
    fprintf(stderr, "mmio_read: %s is %lld bytes; END is past it\n", argv[1], (long long)st.st_size);
    return 2;
  }
  void* p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) {
    fprintf(stderr, "mmio_read: mmap %s: %s\n", argv[1], strerror(errno));
    return 1;
  }
  const volatile uint32_t* bar = (const volatile uint32_t*)p;
  setvbuf(stdout, NULL, _IOLBF, 0);
  for (uint64_t off = start; off < end; off += stride)
    printf("0x%06llx 0x%08x\n", (unsigned long long)off, bar[off / 4]);
  munmap(p, (size_t)st.st_size);
  close(fd);
  return 0;
}
