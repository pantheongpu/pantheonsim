// Decoder probe: VABSDIFF, the 32-bit sum of absolute differences __sad and __usad compile to for sm_86 (signed, .U32, and
// with an immediate as b or as c). Compiled into the SASS corpus.
__global__ void k(const unsigned* in, unsigned* out) {
  const unsigned a = in[0], b = in[1], c = in[2];
  out[0] = (unsigned)__sad((int)a, (int)b, c);
  out[1] = __usad(a, b, c);
  out[2] = __usad(a, 12345u, 7u);
  out[3] = (unsigned)__sad((int)a, -77, b);
}
