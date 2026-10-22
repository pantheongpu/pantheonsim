# Running a driver-API program

A program written against the CUDA driver API links the simulator's driver library (`libvgpucuda`, built as `libcuda.so.1`) directly, with no `nvcc` involved. This is the smallest way to use it. Programs built with `nvcc` take the other route, `vgpu run` (see [Run your own program](getting-started.md#run-your-own-program)).

```c
#include <vgpu_cuda.h>   /* clean-room subset header, in nvidia/include/ */
/* cuInit, cuDeviceGet, cuCtxCreate, cuMemAlloc, cuMemcpyHtoD,
   cuModuleLoadData(ptx_text), cuModuleGetFunction, cuLaunchKernel, ... */
```

```
cc app.c -I<repo>/nvidia/include -L<repo>/build -lcuda -o app
LD_LIBRARY_PATH=<repo>/build VGPU_GPU=nvidia/h200 VGPU_DEVICE_COUNT=4 ./app
```

The environment variables that choose the device and tune the run are in [cli.md](cli.md#environment-variables).
