// A compressing H.264 encoder for VirtualGPU's NVENC: intra pictures (Intra16x16 and Intra4x4
// prediction, the 4x4 integer transform, quantisation, CAVLC) and P pictures (one reference,
// quarter-sample motion search, P_Skip), with a rate-distortion choice between the modes. It writes
// Baseline-tool streams (one slice, CAVLC, no B pictures, no 8x8 transform) that every H.264 decoder
// reads: nvidia/tests/e2e/nvenc_h264.cpp decodes them with ffmpeg and with the NVDEC of this tree.
//
// It is not NVIDIA's encoder and does not try to produce NVIDIA's bytes. Which bitstream a given
// picture becomes is a pure function of the picture, the picture type, the QP, and (for P pictures)
// the previous picture: an IDR picture of the same frame at the same QP is the same bytes every time.
//
// Written from ITU-T H.264 (03/2009): clause 7 for the syntax, 8.3 for intra prediction, 8.4.2.2 for
// the interpolation, 8.5 for the transform, 9.2 for CAVLC. The reference picture a P picture predicts
// from is the decoded, deblocked picture, made by the decoder of this tree (h264_decode.cpp) from the
// bytes just written, so the encoder and any decoder cannot disagree about it.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vgpu_nvenc {

// An 8-bit 4:2:0 picture: luma w x h, chroma ((w+1)/2) x ((h+1)/2).
struct EncPicture {
  int w = 0, h = 0;
  std::vector<uint8_t> y, u, v;
};

enum class PicType { kIdr, kIntra, kInter };

struct EncStats {
  int qp = 0;
  size_t bytes = 0;
  int intra_mbs = 0, inter_mbs = 0, skipped_mbs = 0;
  double psnr_y = 0;   // against the picture as given, after the loop filter
};

class H264Encoder {
 public:
  // profile_idc: 66 (Baseline), 77 (Main) or 100 (High); the tools used are the same in all three.
  // `field_pictures`: interlaced coding with field pictures, for test streams (frames go through encode_field_pair(); the card's
  // NVENC refuses field encoding, so no API path reaches this). `height` is then the frame's, and a multiple of 4.
  H264Encoder(int width, int height, int fps_num, int fps_den, int profile_idc, bool deblock, bool field_pictures = false);
  ~H264Encoder();

  std::vector<uint8_t> parameter_sets() const;   // Annex B: SPS then PPS

  // Encodes one picture and returns its slice NAL unit (Annex B, four-byte start code). A P picture
  // with no reference picture to predict from (the first, or after reset()) becomes an IDR picture.
  // `qp` is the slice QP, 0..51. `stats` may be null.
  std::vector<uint8_t> encode(const EncPicture& in, PicType type, int qp, EncStats* stats = nullptr);

  // Field coding only: codes a frame as two field pictures (the first an IDR, intra or P picture, the second a P picture that may
  // predict from the first). Returns both slice NAL units.
  std::vector<uint8_t> encode_field_pair(const EncPicture& frame, bool top_field_first, PicType type, int qp, EncStats* stats = nullptr);

  // Forget the reference picture and start a new coded video sequence at the next picture.
  void reset();

  // For tests: the reconstruction of the last picture before the loop filter (equal to the decoded
  // picture when the encoder was built with deblock == false), as coded_width() x coded_height()
  // luma samples (macroblock multiples) and the two chroma planes.
  int coded_width() const;
  int coded_height() const;
  const std::vector<uint8_t>& recon_y() const;
  const std::vector<uint8_t>& recon_u() const;
  const std::vector<uint8_t>& recon_v() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

// A QP for the first picture of a stream given a bit rate: from the bits available per sample.
int initial_qp_for(int width, int height, int fps_num, int fps_den, long bitrate);

}  // namespace vgpu_nvenc
