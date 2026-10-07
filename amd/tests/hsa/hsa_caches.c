/* HSA 1.1's cache, wavefront and ISA-compatibility queries, and signal
 * groups, as the specification defines them (ROCm's hsa.h declares the same):
 * a GPU agent's caches in ascending level, their sizes the ones the deprecated
 * HSA_AGENT_INFO_CACHE_SIZE array holds; a traversal that stops where its
 * callback says; one wavefront of the agent's size; and a wait on several
 * signals that returns the one that satisfied its condition. Built against
 * VirtualGPU's HSA header or ROCm's (amd/tests/e2e/run_hsa.sh). One line a
 * check, "ok <what>" or "FAIL <what>". */
#define _DEFAULT_SOURCE
#ifdef VGPU_REAL_HSA
#include <hsa/hsa.h>
#else
#include "vgpu/hsa_abi.h"
#endif

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int failures;
static void check(const char* what, int ok) {
  printf("%s %s\n", ok ? "ok" : "FAIL", what);
  fflush(stdout);
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

typedef struct {
  hsa_agent_t cpu, gpu;
  int cpus, gpus;
} Agents;
static hsa_status_t each_agent(hsa_agent_t a, void* data) {
  Agents* all = (Agents*)data;
  hsa_device_type_t type;
  MUST(hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &type));
  if (type == HSA_DEVICE_TYPE_CPU) all->cpu = a, all->cpus++;
  else if (type == HSA_DEVICE_TYPE_GPU && !all->gpus++) all->gpu = a;
  return HSA_STATUS_SUCCESS;
}

typedef struct {
  int count;
  uint8_t levels[8];
  uint32_t sizes[8];
  int names_ok;
} Caches;
static hsa_status_t each_cache(hsa_cache_t c, void* data) {
  Caches* all = (Caches*)data;
  uint32_t name_length = 0, size = 0;
  uint8_t level = 0;
  MUST(hsa_cache_get_info(c, HSA_CACHE_INFO_NAME_LENGTH, &name_length));
  char name[64] = {0};
  if (name_length >= sizeof name) return HSA_STATUS_ERROR;
  MUST(hsa_cache_get_info(c, HSA_CACHE_INFO_NAME, name));
  MUST(hsa_cache_get_info(c, HSA_CACHE_INFO_LEVEL, &level));
  MUST(hsa_cache_get_info(c, HSA_CACHE_INFO_SIZE, &size));
  if (strlen(name) != name_length) all->names_ok = 0;
  if (all->count < 8) all->levels[all->count] = level, all->sizes[all->count] = size;
  all->count++;
  return HSA_STATUS_SUCCESS;
}
static hsa_status_t stop_at_first(hsa_cache_t c, void* data) {
  (void)c;
  ++*(int*)data;
  return HSA_STATUS_INFO_BREAK;
}
static hsa_status_t one_isa(hsa_isa_t isa, void* data) {
  *(hsa_isa_t*)data = isa;
  return HSA_STATUS_SUCCESS;
}
typedef struct {
  int count;
  uint32_t size;
} Waves;
static hsa_status_t each_wave(hsa_wavefront_t w, void* data) {
  Waves* all = (Waves*)data;
  MUST(hsa_wavefront_get_info(w, HSA_WAVEFRONT_INFO_SIZE, &all->size));
  all->count++;
  return HSA_STATUS_SUCCESS;
}

typedef struct {
  hsa_signal_t signal;
  hsa_signal_value_t value;
} Delayed;
static void* set_later(void* arg) {
  Delayed* d = (Delayed*)arg;
  struct timespec ts = {0, 50 * 1000 * 1000};
  nanosleep(&ts, NULL);
  hsa_signal_store_relaxed(d->signal, d->value);
  return NULL;
}

int main(void) {
  MUST(hsa_init());
  Agents agents = {{0}, {0}, 0, 0};
  MUST(hsa_iterate_agents(each_agent, &agents));

  /* Caches. */
  uint32_t array[4] = {0};
  MUST(hsa_agent_get_info(agents.gpu, HSA_AGENT_INFO_CACHE_SIZE, array));
  Caches caches = {0, {0}, {0}, 1};
  MUST(hsa_agent_iterate_caches(agents.gpu, each_cache, &caches));
  int match = caches.count > 0 && caches.count <= 4;
  for (int i = 0; i < caches.count && i < 4; ++i)
    if (caches.levels[i] != i + 1 || caches.sizes[i] != array[i]) match = 0;
  check("a GPU's caches come in ascending level, with the sizes of HSA_AGENT_INFO_CACHE_SIZE", match);
  check("a cache's name is as long as its name length says", caches.names_ok);
  Caches none = {0, {0}, {0}, 1};
  MUST(hsa_agent_iterate_caches(agents.cpu, each_cache, &none));
  check("the CPU agent has no cache to report", none.count == 0);
  int visited = 0;
  check("a callback's status stops the traversal and is returned",
        hsa_agent_iterate_caches(agents.gpu, stop_at_first, &visited) == HSA_STATUS_INFO_BREAK && visited == 1);
  hsa_cache_t bogus = {0};
  uint32_t scratch = 0;
  check("a cache that is not one, and a NULL callback, are refused",
        hsa_cache_get_info(bogus, HSA_CACHE_INFO_SIZE, &scratch) == HSA_STATUS_ERROR_INVALID_CACHE &&
            hsa_agent_iterate_caches(agents.gpu, NULL, NULL) == HSA_STATUS_ERROR_INVALID_ARGUMENT);

  /* Wavefronts and compatibility. */
  hsa_isa_t isa = {0};
  MUST(hsa_agent_iterate_isas(agents.gpu, one_isa, &isa));
  uint32_t wave = 0;
  MUST(hsa_agent_get_info(agents.gpu, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wave));
  Waves waves = {0, 0};
  MUST(hsa_isa_iterate_wavefronts(isa, each_wave, &waves));
  check("an ISA has one wavefront, of the agent's size", waves.count == 1 && waves.size == wave);
  bool same = false;
  MUST(hsa_isa_compatible(isa, isa, &same));
  check("code for an ISA is compatible with that ISA", same);

  /* Signal groups. */
  hsa_signal_t signals[3];
  for (int i = 0; i < 3; ++i) MUST(hsa_signal_create(1, 0, NULL, &signals[i]));
  hsa_signal_group_t group;
  MUST(hsa_signal_group_create(3, signals, 1, &agents.gpu, &group));
  hsa_signal_condition_t conds[3] = {HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_EQ, HSA_SIGNAL_CONDITION_LT};
  hsa_signal_value_t want[3] = {0, 0, 1};
  hsa_signal_t got = {0};
  hsa_signal_value_t value = 99;
  /* Signal 2 holds 1 and wants less than 1: signal 1 is set to 0 after a wait has begun. */
  Delayed delayed = {signals[1], 0};
  pthread_t thread;
  pthread_create(&thread, NULL, set_later, &delayed);
  hsa_status_t st = hsa_signal_group_wait_any_scacquire(group, conds, want, HSA_WAIT_STATE_BLOCKED, &got, &value);
  pthread_join(thread, NULL);
  check("a wait on a group returns the signal that came to satisfy its condition",
        st == HSA_STATUS_SUCCESS && got.handle == signals[1].handle && value == 0);
  got.handle = 0;
  hsa_signal_store_relaxed(signals[0], 0);
  st = hsa_signal_group_wait_any_relaxed(group, conds, want, HSA_WAIT_STATE_ACTIVE, &got, &value);
  check("a signal already satisfied is returned at once (any of those that are)",
        st == HSA_STATUS_SUCCESS && (got.handle == signals[0].handle || got.handle == signals[1].handle) && value == 0);
  check("a group wait without its arrays is refused",
        hsa_signal_group_wait_any_scacquire(group, NULL, want, HSA_WAIT_STATE_BLOCKED, &got, &value) ==
                HSA_STATUS_ERROR_INVALID_ARGUMENT &&
            hsa_signal_group_create(0, signals, 1, &agents.gpu, &group) == HSA_STATUS_ERROR_INVALID_ARGUMENT);
  MUST(hsa_signal_group_destroy(group));
  check("a destroyed group is not a group",
        hsa_signal_group_destroy(group) == HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP &&
            hsa_signal_group_wait_any_relaxed(group, conds, want, HSA_WAIT_STATE_BLOCKED, &got, &value) ==
                HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP);
  for (int i = 0; i < 3; ++i) MUST(hsa_signal_destroy(signals[i]));
  MUST(hsa_shut_down());
  return failures ? 1 : 0;
}
