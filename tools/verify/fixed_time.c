/* LD_PRELOAD library for tools/verify: pins time() and clock() so programs that seed their random
 * numbers from them (srand(time(NULL)), thrust::default_random_engine(clock())) give the same input
 * on every run. Only for the programs the verification found to differ from themselves; it is not used
 * for the other runs. clock_gettime is left alone (the simulator and the programs' timers use it). */
#include <time.h>
time_t time(time_t* t) {
  const time_t fixed = 1700000000;
  if (t) *t = fixed;
  return fixed;
}
clock_t clock(void) { return 12345; }
