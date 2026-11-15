/* A stand-in for libamd_comgr.so, loaded by tests/hiprtc/fake_hiprtc.cpp: all it has is a version, which the fake compile puts
 * in its output. Built twice (FAKE_COMGR_VERSION), so that the test can swap the compiler library for another one. */
#ifndef FAKE_COMGR_VERSION
#define FAKE_COMGR_VERSION "fake comgr 3.0.1"
#endif
__attribute__((visibility("default"))) const char* fake_comgr_version(void) { return FAKE_COMGR_VERSION; }
