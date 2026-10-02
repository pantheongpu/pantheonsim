// Decoder probe: fmin/fmax on doubles (DSETP.MIN/MAX). Every architecture.
__global__ void k(double* o, const double* a, const double* b) {
  int i = threadIdx.x;
  o[2*i] = fmax(a[i], b[i]); o[2*i+1] = fmin(a[i], b[i]);
}
