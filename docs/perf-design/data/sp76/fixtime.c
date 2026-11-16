/* Round sp76: a time() that answers what FIXTIME says, for programs that seed their random numbers from the clock (Rodinia particlefilter:
 * seed[i] = time(0) * i). Built as a shared object and named in LD_PRELOAD; the program's binary and its kernels are untouched.
 *   cc -shared -fPIC -O1 -o libfixtime.so fixtime.c */
#include <stdlib.h>
#include <time.h>
time_t time(time_t* t) {
  const char* v = getenv("FIXTIME");
  const time_t r = v && *v ? (time_t)atoll(v) : (time_t)1700000000;
  if (t) *t = r;
  return r;
}
