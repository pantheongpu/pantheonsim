/* The HIP version the runtime reports, from a program that was linked (or not) against
 * libamdhip64.so.6 and .so.7. Declared by hand: the program needs no HIP header. */
#include <stdio.h>

int hipRuntimeGetVersion(int*);
int hipDriverGetVersion(int*);

int main(void) {
  int runtime = 0, driver = 0;
  hipRuntimeGetVersion(&runtime);
  hipDriverGetVersion(&driver);
  printf("runtime %d driver %d\n", runtime, driver);
  return runtime == driver ? 0 : 1;
}
