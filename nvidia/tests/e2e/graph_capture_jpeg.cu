// nvJPEG's decode inside a captured CUDA graph (see graph_capture_common.h): the bitstream is parsed at
// the call -- the program frees its copy right after -- and the pixels are written at each launch of the
// graph. The encoders cannot be captured: they wait for the stream, fail, and invalidate the capture.
// run_graph_capture.sh jpeg nvjpeg --card runs the same program on NVIDIA's nvJPEG.
#include <nvjpeg.h>

#include "graph_capture_common.h"

using namespace gc;

#define OK(x) do { nvjpegStatus_t s_ = (x); if (s_ != NVJPEG_STATUS_SUCCESS) { \
  std::printf("     %s -> %d\n", #x, (int)s_); ok = false; } } while (0)

int main() {
  Runner r;
  nvjpegHandle_t h;
  nvjpegCreateSimple(&h);
  nvjpegJpegState_t state;
  nvjpegJpegStateCreate(h, &state);
  const int W = 32, H = 32;
  const size_t px = (size_t)W * H * 3;
  // A JPEG to decode: an image encoded eagerly.
  std::vector<unsigned char> jpeg;
  {
    nvjpegEncoderState_t es;
    nvjpegEncoderParams_t ep;
    nvjpegEncoderStateCreate(h, &es, r.st);
    nvjpegEncoderParamsCreate(h, &ep, r.st);
    nvjpegEncoderParamsSetSamplingFactors(ep, NVJPEG_CSS_444, r.st);
    unsigned char* d = r.alloc<unsigned char>(px);
    std::vector<unsigned char> img(px);
    for (size_t i = 0; i < px; ++i) img[i] = static_cast<unsigned char>((i * 7) % 251);
    cudaMemcpy(d, img.data(), px, cudaMemcpyHostToDevice);
    nvjpegImage_t src{};
    src.channel[0] = d, src.pitch[0] = W * 3;
    nvjpegEncodeImage(h, es, ep, &src, NVJPEG_INPUT_RGBI, W, H, r.st);
    size_t len = 0;
    nvjpegEncodeRetrieveBitstream(h, es, nullptr, &len, r.st);
    jpeg.resize(len);
    nvjpegEncodeRetrieveBitstream(h, es, jpeg.data(), &len, r.st);
    cudaStreamSynchronize(r.st);
    nvjpegEncoderParamsDestroy(ep);
    nvjpegEncoderStateDestroy(es);
  }
  unsigned char *out = r.alloc<unsigned char>(px), *planar = r.alloc<unsigned char>(px);
  r.run("nvjpegDecode (interleaved RGB; the host bitstream freed after the call)", [&] {
    bool ok = true;
    const std::vector<unsigned char> mine = jpeg;   // gone when this returns
    nvjpegImage_t dst{};
    dst.channel[0] = out, dst.pitch[0] = W * 3;
    OK(nvjpegDecode(h, state, mine.data(), mine.size(), NVJPEG_OUTPUT_RGBI, &dst, r.st));
    return ok;
  }, {{out, px, Dt::Bytes}});
  r.run("nvjpegDecode (planar RGB)", [&] {
    bool ok = true;
    const std::vector<unsigned char> mine = jpeg;
    nvjpegImage_t dst{};
    for (int c = 0; c < 3; ++c) dst.channel[c] = planar + c * W * H, dst.pitch[c] = W;
    OK(nvjpegDecode(h, state, mine.data(), mine.size(), NVJPEG_OUTPUT_RGB, &dst, r.st));
    return ok;
  }, {{planar, px, Dt::Bytes}});
  // The encoder, in a capture, fails and invalidates it.
  {
    nvjpegEncoderState_t es;
    nvjpegEncoderParams_t ep;
    nvjpegEncoderStateCreate(h, &es, r.st);
    nvjpegEncoderParamsCreate(h, &ep, r.st);
    nvjpegEncoderParamsSetSamplingFactors(ep, NVJPEG_CSS_444, r.st);
    nvjpegImage_t src{};
    src.channel[0] = out, src.pitch[0] = W * 3;
    // Used once eagerly first, so the capture finds its buffers made (a first encode allocates, and in a
    // capture that fails with ALLOCATOR_FAILURE on the card).
    nvjpegEncodeImage(h, es, ep, &src, NVJPEG_INPUT_RGBI, W, H, r.st);
    cudaStreamSynchronize(r.st);
    cudaGraph_t g = nullptr;
    cudaStreamBeginCapture(r.st, cudaStreamCaptureModeGlobal);
    const nvjpegStatus_t rc = nvjpegEncodeImage(h, es, ep, &src, NVJPEG_INPUT_RGBI, W, H, r.st);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(r.st, &status);
    const cudaError_t e = cudaStreamEndCapture(r.st, &g);
    if (g) cudaGraphDestroy(g);
    cudaGetLastError();
    expect("nvjpegEncodeImage in a capture: EXECUTION_FAILED, and the capture is invalidated",
           rc == NVJPEG_STATUS_EXECUTION_FAILED && status == cudaStreamCaptureStatusInvalidated && e == cudaErrorStreamCaptureInvalidated, rc);
    nvjpegEncoderParamsDestroy(ep);
    nvjpegEncoderStateDestroy(es);
  }
  nvjpegJpegStateDestroy(state);
  nvjpegDestroy(h);
  return finish();
}
