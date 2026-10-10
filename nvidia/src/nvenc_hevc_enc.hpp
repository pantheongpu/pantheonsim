// A compressing HEVC (H.265) encoder for VirtualGPU's NVENC, written from ITU-T H.265 (v1): clause 7 for the syntax, 8.4.4.2 for intra
// prediction, 8.6 for scaling and transformation, 9.3 for CABAC. Coding tree blocks of 32x32 split into coding units of 32, 16 and 8
// samples; intra prediction with all 35 modes (planar, DC, 33 angular), the 4x4 DST and the 4x4 to 32x32 DCTs with quantisation, a
// transform tree of up to four levels, and CABAC with every context its syntax elements use. The mode, the split and the transform tree are
// chosen by rate-distortion cost, with the bits taken from the arithmetic coder's own context states.
//
// It is not NVIDIA's encoder and does not try to produce NVIDIA's bytes. What a picture becomes is a pure function of the picture, its type,
// the QP and the pictures before it: the same IDR picture at the same QP is the same bytes every time.
//
// The encoder keeps its own reconstruction, including the deblocking filter (sample adaptive offset is off), so what a decoder returns is exactly
// that reconstruction. nvidia/tests/unit/test_nvenc_hevc_enc.cpp checks that against an HEVC syntax
// decoder of its own, and nvidia/tests/e2e/nvenc_h264.cpp against ffmpeg's.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "nvenc_encoder.hpp"

namespace vgpu_nvenc {

struct HevcOptions {
  int num_ref = 1;   // reference pictures a P picture may use (1 .. 4)
  int max_b = 0;     // B pictures between P pictures (0: none)
  bool deblock = true;   // the deblocking filter (8.7.2); the reconstruction, and so every reference picture, is the filtered one
};

class HevcEncoder : public VideoEncoder {
 public:
  HevcEncoder(int width, int height, int fps_num, int fps_den);
  HevcEncoder(int width, int height, int fps_num, int fps_den, const HevcOptions& opt);
  ~HevcEncoder() override;

  std::vector<uint8_t> parameter_sets() const override;   // VPS, SPS, PPS
  std::vector<uint8_t> encode(const EncPicture& in, PicType type, int qp, EncStats* stats = nullptr) override;
  void reset() override;
  void rollback() override;
  bool supports_b() const override;
  std::vector<uint8_t> encode_at(const EncPicture& in, PicType type, int qp, int poc, EncStats* stats = nullptr) override;

  // For tests: the reconstruction of the last picture (what a decoder returns), as coded_width() x coded_height() luma samples (multiples of
  // eight) and the two chroma planes.
  int coded_width() const;
  int coded_height() const;
  const std::vector<uint8_t>& recon_y() const;
  const std::vector<uint8_t>& recon_u() const;
  const std::vector<uint8_t>& recon_v() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace vgpu_nvenc
