// cudss_kkt.cpp (SCS's calls, every index and value width) built against cuDSS 1.0: the program needs
// libcudss.so.1, which is what an SCS built with NVIDIA's cuDSS 1.0 needs. run_lib_check.sh links it
// with -l:libcudss.so.1 for the "cudss1" library.
#define VGPU_CUDSS_E2E_1_0
#include "cudss_kkt.cpp"
