// Device printf against an RTX 3060's own output, conversion by conversion.
// The card formats on the host with glibc's rules -- "(nil)", "(null)", hex
// floats -- applies width and precision to %s and %c as to the numbers, takes a
// `*` width from the arguments, and prints a conversion it does not know as
// written without consuming an argument. Only formats the card prints reliably
// are here: it reads an hh argument from the wrong place, prints "%5%%" as
// written, and misreads the arguments after a `*` precision or a size_t
// length, none of which is worth imitating.
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cuda_runtime.h>

__global__ void k(const char* s, const wchar_t* w) {
  printf("a %a %A %.2a %10.1a %a|\n", 3.0, -0.1, 1.0, 2.5, 0.0);
  printf("b %p %p %-12p|\n", (void*)0, (void*)0x1234, (void*)0x1234);
  printf("c %.2s|%8s|%-8s|%8.3s|%s|\n", s, s, s, s, (const char*)0);
  printf("d %5c|%-5c|%c|\n", 'x', 'y', 65);
  printf("e %*d|%-*d|%0*d|\n", 5, 1, 5, 2, 5, 3);
  printf("f %k %d %q|\n", 7);
  printf("i %hd %hu|\n", 70000, 70000);
  printf("j %lld %llx %ld %lu|\n", -1234567890123LL, -1234567890123LL, -5L, 5UL);
  printf("k %+d % d %-6d| %06d %#x %#o %.0f %#.0f|\n", 300, 300, 300, 300, 300, 300, 3.14, 3.14);
  printf("l %g %G %e %E %f %F|\n", 3.14159265358979, 3.14159265358979e20, 3.14, 3.14, 2.5, -2.5);
  printf("m %f %f %e|\n", 1.0 / 0.0, -1.0 / 0.0, -0.0);
  printf("n %lf %i %o %X %u|\n", 2.5, -300, 300, 300, -300);
  // %ls prints the string as narrow bytes up to the first NUL, %lc as %c, and %n as written.
  // One wide argument to a line: several in one make the card misread the rest.
  printf("o %ls|\n", w);
  printf("o2 %5ls|\n", w);
  printf("o3 %.1ls|\n", w);
  printf("o4 %ls|\n", (const wchar_t*)0);
  printf("p %lc|\n", (wint_t)65);
  printf("p2 %3lc|\n", (wint_t)66);
  printf("q %d%n|\n", 3, (int*)0);
}

int main() {
  char* s;
  cudaMallocManaged(&s, 16);
  std::strcpy(s, "hello");
  wchar_t* w;
  cudaMallocManaged(&w, 16 * sizeof(wchar_t));
  w[0] = L'h'; w[1] = L'i'; w[2] = L'!'; w[3] = 0;
  k<<<1, 1>>>(s, w);
  const cudaError_t e = cudaDeviceSynchronize();
  cudaFree(s);
  cudaFree(w);
  if (e != cudaSuccess) {
    std::printf("launch failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  return 0;
}
