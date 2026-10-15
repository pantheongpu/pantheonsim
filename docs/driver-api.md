# Running a driver-API program

A program written against the CUDA driver API links the simulator's `libvgpucuda.so` directly, with no `nvcc` involved. This is the smallest way to use it. Programs built with `nvcc` take the other route, `vgpu run` (see the [README](../README.md#run-your-own-program)).

```c
#include <vgpu_cuda.h>   /* clean-room subset header, in include/ */
/* cuInit, cuDeviceGet, cuCtxCreate, cuMemAlloc, cuMemcpyHtoD,
   cuModuleLoadData(ptx_text), cuModuleGetFunction, cuLaunchKernel, ... */
```

```
cc app.c -I<repo>/include -L<repo>/build -lvgpucuda -o app
VGPU_GPU=nvidia/h200 VGPU_DEVICE_COUNT=4 ./app
```

The environment variables that choose the device and tune the run are in [cli.md](cli.md#environment-variables).
