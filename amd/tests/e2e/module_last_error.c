/* A failing module or function call is the thread's last error, as in ROCm's HIP, where every API returns
 * through HIP_RETURN (clr hipamd/src/hip_internal.hpp, HIP_UPDATE_ERROR_STATE): hipGetLastError gives it
 * back once and then hipSuccess, hipPeekAtLastError gives it without clearing it, and a call that succeeds
 * leaves it where it is (pantheonsim #635). OpenMM's HIP kernel cache tries hipModuleLoad on a file that may
 * not be there and reads hipGetLastError after a miss, expecting hipErrorFileNotFound.
 * Declared by hand: the program needs no HIP header. argv[1] is a gfx942 code object, argv[2] an empty file.
 *
 *  - hipModuleLoad: a missing file (hipErrorFileNotFound), an empty file (hipErrorInvalidImage);
 *  - hipModuleLoadData and hipModuleLoadDataEx on bytes that are no code object (hipErrorInvalidImage);
 *  - hipModuleGetFunction and hipModuleGetGlobal: a name the module lacks (hipErrorNotFound), a null module
 *    (hipErrorInvalidResourceHandle, after the out-pointer and name checks, as ROCm orders them);
 *  - hipModuleUnload: a null module (hipErrorInvalidResourceHandle), one that is not loaded (hipErrorNotFound);
 *  - hipLinkCreate, hipLinkAddData, hipLinkComplete, hipLinkDestroy refusing their arguments. */
#include <stdio.h>
#include <stdlib.h>

typedef void* Module;
typedef void* Function;
int hipModuleLoad(Module*, const char*);
int hipModuleLoadData(Module*, const void*);
int hipModuleLoadDataEx(Module*, const void*, unsigned, void*, void**);
int hipModuleUnload(Module);
int hipModuleGetFunction(Function*, Module, const char*);
int hipModuleGetGlobal(void**, unsigned long*, Module, const char*);
int hipLinkCreate(unsigned, void*, void**, void**);
int hipLinkAddData(void*, int, void*, unsigned long, const char*, unsigned, void*, void**);
int hipLinkComplete(void*, void**, unsigned long*);
int hipLinkDestroy(void*);
int hipGetLastError(void);
int hipPeekAtLastError(void);
enum {
  kSuccess = 0,
  kInvalidValue = 1,
  kInvalidImage = 200,
  kFileNotFound = 301,
  kInvalidHandle = 400, /* hipErrorInvalidHandle and hipErrorInvalidResourceHandle */
  kNotFound = 500
};

static int failed = 0;
static void expect(const char* what, int got, int want) {
  if (got == want) printf("ok    %s\n", what);
  else { printf("FAIL  %s: %d, expected %d\n", what, got, want); failed = 1; }
}
/* The call returned `code`; the last error is that code, once, and then hipSuccess. */
static void expect_recorded(const char* what, int got, int code) {
  char line[200];
  expect(what, got, code);
  snprintf(line, sizeof line, "%s: hipGetLastError", what);
  expect(line, hipGetLastError(), code);
  snprintf(line, sizeof line, "%s: hipGetLastError again", what);
  expect(line, hipGetLastError(), kSuccess);
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <gfx942 code object> <empty file>\n", argv[0]); return 2; }
  Module m = NULL, good = NULL;
  Function f = NULL;
  void* p = NULL;
  unsigned long n = 0;

  expect("no error to begin with", hipGetLastError(), kSuccess);
  expect_recorded("hipModuleLoad of a missing file", hipModuleLoad(&m, "/nonexistent/pantheonsim-635.co"), kFileNotFound);
  expect_recorded("hipModuleLoad of an empty file", hipModuleLoad(&m, argv[2]), kInvalidImage);

  /* Peek reads without clearing; a later call that succeeds leaves the error where it is. */
  expect("hipModuleLoad of a missing file, again", hipModuleLoad(&m, "/nonexistent/pantheonsim-635.co"), kFileNotFound);
  expect("hipPeekAtLastError", hipPeekAtLastError(), kFileNotFound);
  expect("hipPeekAtLastError again", hipPeekAtLastError(), kFileNotFound);
  expect("hipModuleLoad of a code object", hipModuleLoad(&good, argv[1]), kSuccess);
  expect("a success leaves the last error: hipGetLastError", hipGetLastError(), kFileNotFound);
  expect("hipGetLastError then", hipGetLastError(), kSuccess);

  static const char junk[64] = "this is not a code object, nor an offload bundle";
  expect_recorded("hipModuleLoadData of bytes that are no code object", hipModuleLoadData(&m, junk), kInvalidImage);
  expect_recorded("hipModuleLoadDataEx of bytes that are no code object", hipModuleLoadDataEx(&m, junk, 0, NULL, NULL), kInvalidImage);
  expect_recorded("hipModuleLoadData with no module pointer", hipModuleLoadData(NULL, junk), kInvalidValue);

  expect_recorded("hipModuleGetFunction of a missing kernel", hipModuleGetFunction(&f, good, "no_such_kernel"), kNotFound);
  expect_recorded("hipModuleGetFunction of a null module", hipModuleGetFunction(&f, NULL, "k"), kInvalidHandle);
  expect_recorded("hipModuleGetFunction with no name", hipModuleGetFunction(&f, NULL, NULL), kInvalidValue);
  expect_recorded("hipModuleGetFunction with an empty name", hipModuleGetFunction(&f, good, ""), kInvalidValue);
  expect_recorded("hipModuleGetGlobal of a missing variable", hipModuleGetGlobal(&p, &n, good, "no_such_variable"), kNotFound);
  expect_recorded("hipModuleGetGlobal of a null module", hipModuleGetGlobal(&p, &n, NULL, "v"), kInvalidHandle);
  expect_recorded("hipModuleGetGlobal with no name", hipModuleGetGlobal(&p, &n, NULL, NULL), kInvalidValue);

  expect_recorded("hipModuleUnload of a null module", hipModuleUnload(NULL), kInvalidHandle);
  expect("hipModuleUnload", hipModuleUnload(good), kSuccess);
  expect_recorded("hipModuleUnload of a module unloaded", hipModuleUnload(good), kNotFound);

  void* link = NULL;
  void* bin = NULL;
  int bogus = 0;
  expect_recorded("hipLinkCreate with no state pointer", hipLinkCreate(0, NULL, NULL, NULL), kInvalidValue);
  expect_recorded("hipLinkAddData to no link", hipLinkAddData(&bogus, 3, (void*)junk, sizeof junk, "x.o", 0, NULL, NULL), kInvalidHandle);
  expect_recorded("hipLinkComplete of no link", hipLinkComplete(&bogus, &bin, &n), kInvalidValue);
  expect_recorded("hipLinkDestroy of no link", hipLinkDestroy(&bogus), kInvalidValue);
  expect("hipLinkCreate", hipLinkCreate(0, NULL, NULL, &link), kSuccess);
  expect("hipLinkDestroy", hipLinkDestroy(link), kSuccess);
  expect("no error at the end", hipGetLastError(), kSuccess);

  puts(failed ? "FAILED" : "PASSED");
  return failed;
}
