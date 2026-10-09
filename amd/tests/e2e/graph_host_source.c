/* A host source of a captured copy: when its bytes are taken. Declared by hand: the program needs no
 * HIP header and no device code.
 *
 *  - heap memory is read when the graph is launched (hip-tests refill the input before each launch);
 *  - a stack variable of a frame that has gone by then is taken as it was at capture (rocPRIM's
 *    histogram copies a local of its own to the device and is launched after it has returned);
 *  - a stack variable of a frame still there is read as it is at launch, as the heap's is. */
#include <stdio.h>
#include <stdlib.h>

typedef void* Stream;
typedef void* Graph;
typedef void* Exec;
int hipMalloc(void**, unsigned long);
int hipStreamCreate(Stream*);
int hipStreamSynchronize(Stream);
int hipStreamBeginCapture(Stream, int);
int hipStreamEndCapture(Stream, Graph*);
int hipGraphInstantiate(Exec*, Graph, void*, char*, unsigned long);
int hipGraphLaunch(Exec, Stream);
int hipMemcpyAsync(void*, const void*, unsigned long, int, Stream);
int hipMemcpy(void*, const void*, unsigned long, int);
enum { kHostToDevice = 1, kDeviceToHost = 2 };

static int failed = 0;
static void expect(const char* what, int got, int want) {
  if (got == want) printf("ok    %s\n", what);
  else { printf("FAIL  %s: %d, expected %d\n", what, got, want); failed = 1; }
}

/* Captures a copy of a local that is gone when it returns. */
__attribute__((noinline)) static void capture_a_local(Stream s, int* dev, int value) {
  volatile int local = value;
  hipMemcpyAsync(dev, (const void*)&local, sizeof local, kHostToDevice, s);
}
/* Overwrites the stack the callee used. */
__attribute__((noinline)) static int scribble(int depth) {
  volatile char pad[256];
  for (int i = 0; i < 256; ++i) pad[i] = (char)(0x5a + depth);
  return depth ? scribble(depth - 1) + pad[depth] : pad[0];
}
static int read_back(int* dev) {
  int v = -1;
  hipMemcpy(&v, dev, sizeof v, kDeviceToHost);
  return v;
}

int main(void) {
  Stream s;
  int *dev, *dev2;
  hipMalloc((void**)&dev, 64);
  hipMalloc((void**)&dev2, 64);
  hipStreamCreate(&s);

  /* The local of a frame that has gone. */
  hipStreamBeginCapture(s, 0);
  capture_a_local(s, dev, 1234);
  scribble(8);
  Graph g;
  hipStreamEndCapture(s, &g);
  Exec x;
  hipGraphInstantiate(&x, g, NULL, NULL, 0);
  hipMemcpy(dev, &(int){0}, sizeof(int), kHostToDevice);
  scribble(8);
  hipGraphLaunch(x, s);
  hipStreamSynchronize(s);
  expect("a local of a frame that has gone is taken as it was at capture", read_back(dev), 1234);

  /* Heap memory refilled before each launch. */
  int* heap = malloc(sizeof(int));
  *heap = 1;
  hipStreamBeginCapture(s, 0);
  hipMemcpyAsync(dev, heap, sizeof(int), kHostToDevice, s);
  hipStreamEndCapture(s, &g);
  hipGraphInstantiate(&x, g, NULL, NULL, 0);
  for (int i = 1; i <= 3; ++i) {
    *heap = 100 * i;
    hipGraphLaunch(x, s);
    hipStreamSynchronize(s);
    char what[64];
    snprintf(what, sizeof what, "heap source read at launch %d", i);
    expect(what, read_back(dev), 100 * i);
  }
  free(heap);

  /* A local of a frame that is still there, refilled before each launch. */
  volatile int mine = 7;
  hipStreamBeginCapture(s, 0);
  hipMemcpyAsync(dev2, (const void*)&mine, sizeof mine, kHostToDevice, s);
  hipStreamEndCapture(s, &g);
  hipGraphInstantiate(&x, g, NULL, NULL, 0);
  for (int i = 1; i <= 3; ++i) {
    mine = 10 * i;
    hipGraphLaunch(x, s);
    hipStreamSynchronize(s);
    char what[64];
    snprintf(what, sizeof what, "live local read at launch %d", i);
    expect(what, read_back(dev2), 10 * i);
  }
  return failed;
}
