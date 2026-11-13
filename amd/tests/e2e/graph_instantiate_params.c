/* hipGraphInstantiateWithParams with each flag ROCm's HIP takes there (pantheonsim #628). Colvars instantiates
 * its graphs with hipGraphInstantiateFlagUpload and an uploadStream, which was refused as an invalid value.
 * Declared by hand: the program needs no HIP header and no device code.
 *
 *  - Upload (2), alone or with AutoFreeOnLaunch (1), on a created stream or the null stream: the graph is
 *    instantiated, result_out is hipGraphInstantiateSuccess, errNode_out is cleared, and the graph runs;
 *  - DeviceLaunch (4) is taken too, as ROCm takes it, and hipGraphExecGetFlags reports the flags given;
 *  - a flag outside the four is refused with hipErrorInvalidValue and result_out hipGraphInstantiateError;
 *  - hipGraphInstantiateWithFlags, which has no stream, still refuses Upload and DeviceLaunch, as ROCm does. */
#include <stdio.h>
#include <stdlib.h>

typedef void* Stream;
typedef void* Graph;
typedef void* Exec;
typedef struct {
  void* errNode_out;
  unsigned long long flags;
  int result_out;
  void* uploadStream;
} InstantiateParams; /* hipGraphInstantiateParams */
int hipMalloc(void**, unsigned long);
int hipStreamCreate(Stream*);
int hipStreamSynchronize(Stream);
int hipStreamBeginCapture(Stream, int);
int hipStreamEndCapture(Stream, Graph*);
int hipGraphInstantiateWithParams(Exec*, Graph, InstantiateParams*);
int hipGraphInstantiateWithFlags(Exec*, Graph, unsigned long long);
int hipGraphExecGetFlags(Exec, unsigned long long*);
int hipGraphExecDestroy(Exec);
int hipGraphUpload(Exec, Stream);
int hipGraphLaunch(Exec, Stream);
int hipMemcpyAsync(void*, const void*, unsigned long, int, Stream);
int hipMemcpy(void*, const void*, unsigned long, int);
enum { kHostToDevice = 1, kDeviceToHost = 2, kDeviceToDevice = 3 };
enum { kSuccess = 0, kInvalidValue = 1 };
enum { kAutoFree = 1, kUpload = 2, kDeviceLaunch = 4, kUseNodePriority = 8 };

static int failed = 0;
static void expect(const char* what, long long got, long long want) {
  if (got == want) printf("ok    %s\n", what);
  else { printf("FAIL  %s: %lld, expected %lld\n", what, got, want); failed = 1; }
}

static int* src;
static int *dev, *dev2;
static int read_back(void) {
  int v = -1;
  hipMemcpy(&v, dev2, sizeof v, kDeviceToHost);
  return v;
}

/* Instantiates with the flags and stream given, then launches the graph twice on s with new input. */
static void instantiate_and_run(Graph g, Stream s, unsigned long long flags, Stream upload_on, const char* name) {
  char what[160];
  InstantiateParams p = {(void*)0x1, flags, -1, upload_on};
  Exec x = NULL;
  snprintf(what, sizeof what, "%s: instantiated", name);
  expect(what, hipGraphInstantiateWithParams(&x, g, &p), kSuccess);
  snprintf(what, sizeof what, "%s: result_out is hipGraphInstantiateSuccess", name);
  expect(what, p.result_out, 0);
  snprintf(what, sizeof what, "%s: errNode_out cleared", name);
  expect(what, p.errNode_out == NULL, 1);
  unsigned long long got = ~0ull;
  snprintf(what, sizeof what, "%s: hipGraphExecGetFlags reports the flags given", name);
  expect(what, hipGraphExecGetFlags(x, &got) == kSuccess ? (long long)got : -1, (long long)flags);
  for (int i = 1; i <= 2; ++i) {
    *src = 1000 * (int)flags + i;
    snprintf(what, sizeof what, "%s: launch %d starts", name, i);
    expect(what, hipGraphLaunch(x, s), kSuccess);
    hipStreamSynchronize(s);
    snprintf(what, sizeof what, "%s: launch %d computes", name, i);
    expect(what, read_back(), 1000 * (int)flags + i);
  }
  hipGraphExecDestroy(x);
}

int main(void) {
  Stream s, other;
  hipMalloc((void**)&dev, 64);
  hipMalloc((void**)&dev2, 64);
  hipStreamCreate(&s);
  hipStreamCreate(&other);
  src = malloc(sizeof(int));
  *src = 0;

  /* The graph copies the host value in and then from one device buffer to another. */
  Graph g;
  hipStreamBeginCapture(s, 0);
  hipMemcpyAsync(dev, src, sizeof(int), kHostToDevice, s);
  hipMemcpyAsync(dev2, dev, sizeof(int), kDeviceToDevice, s);
  expect("captured", hipStreamEndCapture(s, &g), kSuccess);

  instantiate_and_run(g, s, kUpload, s, "Upload on the launch stream (Colvars)");
  instantiate_and_run(g, s, kUpload, other, "Upload on another stream");
  instantiate_and_run(g, s, kUpload | kAutoFree, NULL, "Upload|AutoFreeOnLaunch on the null stream");
  instantiate_and_run(g, s, kUpload | kUseNodePriority, s, "Upload|UseNodePriority");
  instantiate_and_run(g, s, kDeviceLaunch, NULL, "DeviceLaunch");
  instantiate_and_run(g, s, 0, NULL, "no flags");

  InstantiateParams bad = {NULL, 16, -1, s};
  Exec x = NULL;
  expect("a flag outside the four is refused", hipGraphInstantiateWithParams(&x, g, &bad), kInvalidValue);
  expect("with result_out hipGraphInstantiateError", bad.result_out, 1);
  expect("hipGraphInstantiateWithFlags refuses Upload", hipGraphInstantiateWithFlags(&x, g, kUpload), kInvalidValue);
  expect("hipGraphInstantiateWithFlags refuses DeviceLaunch", hipGraphInstantiateWithFlags(&x, g, kDeviceLaunch),
         kInvalidValue);
  expect("hipGraphInstantiateWithFlags takes AutoFreeOnLaunch", hipGraphInstantiateWithFlags(&x, g, kAutoFree),
         kSuccess);
  expect("hipGraphUpload of it", hipGraphUpload(x, s), kSuccess);
  free(src);
  return failed;
}
