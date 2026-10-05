/* LD_PRELOAD shim for suites.sh digest: time() always answers the same second.
 * A program that seeds its random numbers from the clock (CUDA Samples'
 * fp16ScalarProduct: srand(time(NULL))) hands the SASS run and the PTX run,
 * made a second or more apart, different inputs, so the two runs' memory
 * differs from the first upload on and says nothing about the engines. */
#include <time.h>

time_t time(time_t *t) {
  const time_t fixed = 1700000000;
  if (t) *t = fixed;
  return fixed;
}
