/* A stand-in for libhiprtc-builtins.so: the header the real libhiprtc compiles every program with. Built twice
 * (FAKE_BUILTINS_PLAIN: a header that does not define hip_runtime.h's include guard). */
#ifdef FAKE_BUILTINS_PLAIN
__attribute__((visibility("default"))) const char __hipRTC_header[] = "// fake hiprtc_runtime.h, without the guard\n#define FAKE_RUNTIME 2\n";
#else
__attribute__((visibility("default"))) const char __hipRTC_header[] =
    "// fake hiprtc_runtime.h\n#define HIP_INCLUDE_HIP_HIP_RUNTIME_H\n#define FAKE_RUNTIME 1\n";
#endif
__attribute__((visibility("default"))) const unsigned long __hipRTC_header_size = sizeof __hipRTC_header;
