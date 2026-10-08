/* OpenCL on a simulated AMD GPU: ROCm's OpenCL runtime (libamdocl64, the same
 * CLR that HIP is built on), which compiles each kernel with the compiler
 * library (comgr) and runs it through the HSA runtime. Everything it needs of
 * the machine comes from the simulator's HSA layer.
 *
 * Every check prints "ok" or "FAIL"; the program exits non-zero on any failure. */
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failed = 0, g_checks = 0;
#define CHECK(cond, ...)                                  \
  do {                                                    \
    ++g_checks;                                           \
    if (!(cond)) {                                        \
      ++g_failed;                                         \
      printf("FAIL line %d: %s: ", __LINE__, #cond);      \
      printf(__VA_ARGS__);                                \
      printf("\n");                                       \
    }                                                     \
  } while (0)

static const char* kSource =
    "__kernel void vadd(__global const float* a, __global const float* b, __global float* c, int n) {\n"
    "  int i = get_global_id(0);\n"
    "  if (i < n) c[i] = a[i] + b[i];\n"
    "}\n"
    "__kernel void reduce(__global const int* in, __global int* out, __local int* scratch, int n) {\n"
    "  int gid = get_global_id(0), lid = get_local_id(0);\n"
    "  scratch[lid] = gid < n ? in[gid] : 0;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
    "    if (lid < s) scratch[lid] += scratch[lid + s];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  if (lid == 0) out[get_group_id(0)] = scratch[0];\n"
    "}\n"
    "__kernel void histogram(__global const uchar* data, __global volatile int* bins, int n) {\n"
    "  int i = get_global_id(0);\n"
    "  if (i < n) atomic_inc(&bins[data[i]]);\n"
    "}\n"
    "__kernel void dscale(__global double* x, double k, int n) {\n"
    "  int i = get_global_id(0);\n"
    "  if (i < n) x[i] = x[i] * k + sin(0.5 * (double)i) * 0.0;\n"
    "}\n"
    "__kernel void matmul(__global const float* a, __global const float* b, __global float* c, int m, int n, int k) {\n"
    "  int row = get_global_id(1), col = get_global_id(0);\n"
    "  if (row >= m || col >= n) return;\n"
    "  float s = 0;\n"
    "  for (int p = 0; p < k; ++p) s += a[row * k + p] * b[p * n + col];\n"
    "  c[row * n + col] = s;\n"
    "}\n"
    "__kernel void prefix(__global int* x, int n) {\n"
    "  /* one work-group, Hillis-Steele in global memory with barriers */\n"
    "  int i = get_local_id(0);\n"
    "  for (int d = 1; d < n; d <<= 1) {\n"
    "    int v = i >= d ? x[i - d] : 0;\n"
    "    barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "    if (i < n) x[i] += v;\n"
    "    barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "  }\n"
    "}\n"
    "__kernel void blur(read_only image2d_t src, write_only image2d_t dst, sampler_t smp) {\n"
    "  int x = get_global_id(0), y = get_global_id(1);\n"
    "  float4 s = read_imagef(src, smp, (int2)(x, y)) + read_imagef(src, smp, (int2)(x + 1, y));\n"
    "  write_imagef(dst, (int2)(x, y), s * 0.5f);\n"
    "}\n"
    "__kernel void svm_inc(__global int* p, int n) {\n"
    "  int i = get_global_id(0);\n"
    "  if (i < n) p[i] += 100;\n"
    "}\n"
    "__kernel void offsets(__global int* out) {\n"
    "  int i = get_global_id(0);\n"
    "  out[i - get_global_offset(0)] = i * 10 + (int)get_local_size(0);\n"
    "}\n"
    "__kernel void who(__global int* out) {\n"
    "  if (get_global_id(0) == 0) { out[0] = get_work_dim(); out[1] = get_num_groups(0); out[2] = get_local_size(0); }\n"
    "}\n";

int main(void) {
  cl_int err;
  cl_platform_id platform;
  cl_uint nplat = 0, ndev = 0;
  CHECK(clGetPlatformIDs(1, &platform, &nplat) == CL_SUCCESS && nplat >= 1, "no platform");
  cl_device_id devs[8];
  CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 8, devs, &ndev) == CL_SUCCESS && ndev >= 1, "no GPU device");
  if (g_failed) { printf("%d checks, %d failed\n", g_checks, g_failed); return 1; }
  char name[256] = "", version[256] = "";
  clGetDeviceInfo(devs[0], CL_DEVICE_NAME, sizeof name, name, NULL);
  clGetDeviceInfo(devs[0], CL_DEVICE_VERSION, sizeof version, version, NULL);
  printf("devices %u first %s\n", ndev, name);

  for (cl_uint d = 0; d < ndev; ++d) {
    cl_context ctx = clCreateContext(NULL, 1, &devs[d], NULL, NULL, &err);
    CHECK(err == CL_SUCCESS, "context %d", err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, devs[d], NULL, &err);
    CHECK(err == CL_SUCCESS, "queue %d", err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kSource, NULL, &err);
    err = clBuildProgram(prog, 1, &devs[d], "-cl-std=CL2.0", NULL, NULL);
    if (err != CL_SUCCESS) {
      char log[4096] = "";
      clGetProgramBuildInfo(prog, devs[d], CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
      printf("build log: %s\n", log);
    }
    CHECK(err == CL_SUCCESS, "build %d", err);
    if (err != CL_SUCCESS) continue;

    /* vector add */
    {
      const int n = 100003;
      float *a = malloc(n * 4), *b = malloc(n * 4), *c = malloc(n * 4);
      for (int i = 0; i < n; ++i) { a[i] = (float)(i % 101); b[i] = 0.5f * (float)(i % 17); }
      cl_mem ma = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * 4, a, &err);
      cl_mem mb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * 4, b, &err);
      cl_mem mc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * 4, NULL, &err);
      cl_kernel k = clCreateKernel(prog, "vadd", &err);
      CHECK(err == CL_SUCCESS, "kernel vadd %d", err);
      int nn = n;
      clSetKernelArg(k, 0, sizeof ma, &ma); clSetKernelArg(k, 1, sizeof mb, &mb);
      clSetKernelArg(k, 2, sizeof mc, &mc); clSetKernelArg(k, 3, sizeof nn, &nn);
      size_t global = ((n + 255) / 256) * 256, local = 256;
      CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL) == CL_SUCCESS, "launch vadd");
      CHECK(clEnqueueReadBuffer(q, mc, CL_TRUE, 0, n * 4, c, 0, NULL, NULL) == CL_SUCCESS, "read vadd");
      int wrong = 0;
      for (int i = 0; i < n; ++i) wrong += c[i] != a[i] + b[i];
      CHECK(wrong == 0, "vadd: %d wrong", wrong);
      clReleaseKernel(k); clReleaseMemObject(ma); clReleaseMemObject(mb); clReleaseMemObject(mc);
      free(a); free(b); free(c);
    }

    /* work-group reduction through local memory and barriers */
    {
      const int n = 1 << 14, groups = n / 128;
      int* in = malloc(n * 4);
      int* out = malloc(groups * 4);
      long want = 0, got = 0;
      for (int i = 0; i < n; ++i) { in[i] = (i * 7) % 13 - 6; want += in[i]; }
      cl_mem min = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * 4, in, &err);
      cl_mem mout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, groups * 4, NULL, &err);
      cl_kernel k = clCreateKernel(prog, "reduce", &err);
      int nn = n;
      clSetKernelArg(k, 0, sizeof min, &min); clSetKernelArg(k, 1, sizeof mout, &mout);
      clSetKernelArg(k, 2, 128 * 4, NULL); clSetKernelArg(k, 3, sizeof nn, &nn);
      size_t global = n, local = 128;
      CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL) == CL_SUCCESS, "launch reduce");
      CHECK(clEnqueueReadBuffer(q, mout, CL_TRUE, 0, groups * 4, out, 0, NULL, NULL) == CL_SUCCESS, "read reduce");
      for (int g = 0; g < groups; ++g) got += out[g];
      CHECK(got == want, "reduce: %ld != %ld", got, want);
      clReleaseKernel(k); clReleaseMemObject(min); clReleaseMemObject(mout);
      free(in); free(out);
    }

    /* atomics */
    {
      const int n = 50000;
      unsigned char* data = malloc(n);
      int hist[256] = {0}, want[256] = {0};
      for (int i = 0; i < n; ++i) { data[i] = (unsigned char)((i * 31 + i / 7) & 255); want[data[i]]++; }
      cl_mem md = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n, data, &err);
      cl_mem mh = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof hist, hist, &err);
      cl_kernel k = clCreateKernel(prog, "histogram", &err);
      int nn = n;
      clSetKernelArg(k, 0, sizeof md, &md); clSetKernelArg(k, 1, sizeof mh, &mh); clSetKernelArg(k, 2, sizeof nn, &nn);
      size_t global = ((n + 255) / 256) * 256, local = 256;
      CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL) == CL_SUCCESS, "launch histogram");
      CHECK(clEnqueueReadBuffer(q, mh, CL_TRUE, 0, sizeof hist, hist, 0, NULL, NULL) == CL_SUCCESS, "read histogram");
      int wrong = 0;
      for (int b = 0; b < 256; ++b) wrong += hist[b] != want[b];
      CHECK(wrong == 0, "histogram: %d bins wrong", wrong);
      clReleaseKernel(k); clReleaseMemObject(md); clReleaseMemObject(mh);
      free(data);
    }

    /* doubles */
    {
      const int n = 1000;
      double x[1000];
      for (int i = 0; i < n; ++i) x[i] = 0.25 * i;
      cl_mem mx = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof x, x, &err);
      cl_kernel k = clCreateKernel(prog, "dscale", &err);
      double scale = 3.0;
      int nn = n;
      clSetKernelArg(k, 0, sizeof mx, &mx); clSetKernelArg(k, 1, sizeof scale, &scale); clSetKernelArg(k, 2, sizeof nn, &nn);
      size_t global = 1024;
      CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, NULL, 0, NULL, NULL) == CL_SUCCESS, "launch dscale");
      CHECK(clEnqueueReadBuffer(q, mx, CL_TRUE, 0, sizeof x, x, 0, NULL, NULL) == CL_SUCCESS, "read dscale");
      int wrong = 0;
      for (int i = 0; i < n; ++i) wrong += x[i] != 0.75 * i;
      CHECK(wrong == 0, "double scale: %d wrong", wrong);
      clReleaseKernel(k); clReleaseMemObject(mx);
    }

    /* a 2D range */
    {
      const int m = 37, n = 29, kk = 23;
      float a[37 * 23], b[23 * 29], c[37 * 29], want[37 * 29];
      for (int i = 0; i < m * kk; ++i) a[i] = (float)((i * 5) % 7) - 3;
      for (int i = 0; i < kk * n; ++i) b[i] = (float)((i * 3) % 5) - 2;
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          float s = 0;
          for (int p = 0; p < kk; ++p) s += a[i * kk + p] * b[p * n + j];
          want[i * n + j] = s;
        }
      cl_mem ma = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof a, a, &err);
      cl_mem mb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof b, b, &err);
      cl_mem mc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof c, NULL, &err);
      cl_kernel k = clCreateKernel(prog, "matmul", &err);
      clSetKernelArg(k, 0, sizeof ma, &ma); clSetKernelArg(k, 1, sizeof mb, &mb); clSetKernelArg(k, 2, sizeof mc, &mc);
      clSetKernelArg(k, 3, sizeof m, &m); clSetKernelArg(k, 4, sizeof n, &n); clSetKernelArg(k, 5, sizeof kk, &kk);
      size_t global[2] = {32, 40}, local[2] = {8, 8};
      CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, global, local, 0, NULL, NULL) == CL_SUCCESS, "launch matmul");
      CHECK(clEnqueueReadBuffer(q, mc, CL_TRUE, 0, sizeof c, c, 0, NULL, NULL) == CL_SUCCESS, "read matmul");
      int wrong = 0;
      for (int i = 0; i < m * n; ++i) wrong += c[i] != want[i];
      CHECK(wrong == 0, "matmul: %d of %d wrong", wrong, m * n);
      clReleaseKernel(k); clReleaseMemObject(ma); clReleaseMemObject(mb); clReleaseMemObject(mc);
    }

    /* global-memory barriers within one work-group */
    {
      int x[256], want[256];
      for (int i = 0; i < 256; ++i) { x[i] = i % 5 + 1; want[i] = (i ? want[i - 1] : 0) + x[i]; }
      cl_mem mx = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof x, x, &err);
      cl_kernel k = clCreateKernel(prog, "prefix", &err);
      int nn = 256;
      clSetKernelArg(k, 0, sizeof mx, &mx); clSetKernelArg(k, 1, sizeof nn, &nn);
      size_t global = 256, local = 256;
      CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL) == CL_SUCCESS, "launch prefix");
      CHECK(clEnqueueReadBuffer(q, mx, CL_TRUE, 0, sizeof x, x, 0, NULL, NULL) == CL_SUCCESS, "read prefix");
      int wrong = 0;
      for (int i = 0; i < 256; ++i) wrong += x[i] != want[i];
      CHECK(wrong == 0, "prefix sum: %d wrong", wrong);
      clReleaseKernel(k); clReleaseMemObject(mx);
    }

    /* buffer commands: fill, copy, map, and an event's profiling */
    {
      cl_mem m1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4096, NULL, &err);
      cl_mem m2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4096, NULL, &err);
      int pattern = 0x01020304;
      CHECK(clEnqueueFillBuffer(q, m1, &pattern, sizeof pattern, 0, 4096, 0, NULL, NULL) == CL_SUCCESS, "fill");
      CHECK(clEnqueueCopyBuffer(q, m1, m2, 0, 0, 4096, 0, NULL, NULL) == CL_SUCCESS, "copy");
      int* p = clEnqueueMapBuffer(q, m2, CL_TRUE, CL_MAP_READ, 0, 4096, 0, NULL, NULL, &err);
      CHECK(err == CL_SUCCESS && p, "map %d", err);
      int wrong = 0;
      for (int i = 0; p && i < 1024; ++i) wrong += p[i] != pattern;
      CHECK(wrong == 0, "fill and copy: %d wrong", wrong);
      if (p) clEnqueueUnmapMemObject(q, m2, p, 0, NULL, NULL);
      clFinish(q);
      clReleaseMemObject(m1); clReleaseMemObject(m2);
    }

    /* images, where the device has them (the MI300 family has no texture units) */
    {
      cl_bool images = CL_FALSE;
      clGetDeviceInfo(devs[d], CL_DEVICE_IMAGE_SUPPORT, sizeof images, &images, NULL);
      /* gfx9 image instructions (MIMG) are not implemented: a program is told the device has
       * images, as MI250X does, and VGPU_CL_NO_IMAGES skips what it cannot yet do. */
      if (getenv("VGPU_CL_NO_IMAGES") && getenv("VGPU_CL_NO_IMAGES")[0] == '1') images = CL_FALSE;
      if (images) {
        const int w = 24, h = 10;
        float src[24 * 10 * 4], dst[24 * 10 * 4];
        for (int i = 0; i < w * h * 4; ++i) src[i] = (float)(i % 29);
        cl_image_format fmt = {CL_RGBA, CL_FLOAT};
        cl_image_desc desc;
        memset(&desc, 0, sizeof desc);
        desc.image_type = CL_MEM_OBJECT_IMAGE2D; desc.image_width = w; desc.image_height = h;
        cl_mem is = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &desc, src, &err);
        CHECK(err == CL_SUCCESS, "create source image %d", err);
        cl_mem id = clCreateImage(ctx, CL_MEM_WRITE_ONLY, &fmt, &desc, NULL, &err);
        CHECK(err == CL_SUCCESS, "create destination image %d", err);
        cl_sampler smp = clCreateSamplerWithProperties(ctx, (cl_sampler_properties[]){CL_SAMPLER_NORMALIZED_COORDS, CL_FALSE,
                                                       CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_CLAMP_TO_EDGE,
                                                       CL_SAMPLER_FILTER_MODE, CL_FILTER_NEAREST, 0}, &err);
        CHECK(err == CL_SUCCESS, "sampler %d", err);
        cl_kernel k = clCreateKernel(prog, "blur", &err);
        CHECK(err == CL_SUCCESS, "kernel blur %d", err);
        clSetKernelArg(k, 0, sizeof is, &is); clSetKernelArg(k, 1, sizeof id, &id); clSetKernelArg(k, 2, sizeof smp, &smp);
        size_t global[2] = {w, h};
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, global, NULL, 0, NULL, NULL) == CL_SUCCESS, "launch blur");
        size_t origin[3] = {0, 0, 0}, region[3] = {w, h, 1};
        CHECK(clEnqueueReadImage(q, id, CL_TRUE, origin, region, 0, 0, dst, 0, NULL, NULL) == CL_SUCCESS, "read image");
        int wrong = 0;
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            for (int c = 0; c < 4; ++c) {
              const int x1 = x + 1 < w ? x + 1 : w - 1;
              const float want = 0.5f * (src[(y * w + x) * 4 + c] + src[(y * w + x1) * 4 + c]);
              wrong += dst[(y * w + x) * 4 + c] != want;
            }
        CHECK(wrong == 0, "image read and write: %d wrong", wrong);
        clReleaseKernel(k); clReleaseSampler(smp); clReleaseMemObject(is); clReleaseMemObject(id);
      } else {
        printf("device %u has no image support\n", d);
      }
    }

    /* coarse-grained shared virtual memory */
    {
      cl_device_svm_capabilities svm = 0;
      clGetDeviceInfo(devs[d], CL_DEVICE_SVM_CAPABILITIES, sizeof svm, &svm, NULL);
      CHECK(svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER, "no coarse-grain SVM (%lx)", (unsigned long)svm);
      if (svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
        const int n = 512;
        int* p = clSVMAlloc(ctx, CL_MEM_READ_WRITE, n * sizeof(int), 0);
        CHECK(p != NULL, "clSVMAlloc");
        if (p) {
          CHECK(clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE, p, n * sizeof(int), 0, NULL, NULL) == CL_SUCCESS, "svm map");
          for (int i = 0; i < n; ++i) p[i] = i;
          CHECK(clEnqueueSVMUnmap(q, p, 0, NULL, NULL) == CL_SUCCESS, "svm unmap");
          cl_kernel k = clCreateKernel(prog, "svm_inc", &err);
          clSetKernelArgSVMPointer(k, 0, p);
          int nn = n;
          clSetKernelArg(k, 1, sizeof nn, &nn);
          size_t global = n;
          CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, NULL, 0, NULL, NULL) == CL_SUCCESS, "launch svm_inc");
          CHECK(clEnqueueSVMMap(q, CL_TRUE, CL_MAP_READ, p, n * sizeof(int), 0, NULL, NULL) == CL_SUCCESS, "svm map for read");
          int wrong = 0;
          for (int i = 0; i < n; ++i) wrong += p[i] != i + 100;
          CHECK(wrong == 0, "svm: %d wrong", wrong);
          clEnqueueSVMUnmap(q, p, 0, NULL, NULL);
          clFinish(q);
          clReleaseKernel(k);
          clSVMFree(ctx, p);
        }
      }
    }

    /* a global offset, and a range that is not a multiple of the work-group (OpenCL 2.0) */
    {
      int out[100];
      memset(out, 0, sizeof out);
      cl_mem mo = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof out, out, &err);
      cl_kernel k = clCreateKernel(prog, "offsets", &err);
      clSetKernelArg(k, 0, sizeof mo, &mo);
      size_t offset = 7, global = 90, local = 32;   /* 90 is not a multiple of 32 */
      err = clEnqueueNDRangeKernel(q, k, 1, &offset, &global, &local, 0, NULL, NULL);
      CHECK(err == CL_SUCCESS, "launch with an offset and a ragged range %d", err);
      CHECK(clEnqueueReadBuffer(q, mo, CL_TRUE, 0, sizeof out, out, 0, NULL, NULL) == CL_SUCCESS, "read offsets");
      int wrong = 0;
      for (int i = 0; i < 90; ++i) {
        const int group = (i / 32), size = (group == 2) ? 26 : 32;
        wrong += out[i] != (i + 7) * 10 + size;
      }
      CHECK(wrong == 0, "offset and ragged range: %d wrong", wrong);
      clReleaseKernel(k); clReleaseMemObject(mo);
    }

    /* the work-item functions and the launch dimensions */
    {
      int out[3] = {-1, -1, -1};
      cl_mem mo = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof out, out, &err);
      cl_kernel k = clCreateKernel(prog, "who", &err);
      clSetKernelArg(k, 0, sizeof mo, &mo);
      size_t global = 192, local = 64;
      CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL) == CL_SUCCESS, "launch who");
      CHECK(clEnqueueReadBuffer(q, mo, CL_TRUE, 0, sizeof out, out, 0, NULL, NULL) == CL_SUCCESS, "read who");
      CHECK(out[0] == 1 && out[1] == 3 && out[2] == 64, "work dim %d groups %d local %d", out[0], out[1], out[2]);
      clReleaseKernel(k); clReleaseMemObject(mo);
    }

    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
  }
  printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed ? 1 : 0;
}
