// A compressing H.264 encoder for VirtualGPU's NVENC, written from ITU-T H.264 (03/2009): clause 7 for the syntax, 8.3 for intra prediction, 8.4 for
// inter prediction and the interpolation, 8.5 for the transforms, 9.2 for CAVLC and 9.3 for CABAC.
//
//   * intra pictures: Intra16x16, Intra4x4 and (High profile) Intra8x8 prediction, the 4x4 and 8x8 integer transforms, quantisation;
//   * P pictures: P_Skip and 16x16, 16x8, 8x16 and 8x8 partitions, up to four reference pictures, quarter-sample motion search;
//   * B pictures (Main and High profile): B_Skip and B_Direct_16x16 with spatial direct prediction, 16x16, 16x8, 8x16 and 8x8 partitions predicted from
//     list 0, list 1 or both, and the reordering of pictures for display that B pictures need;
//   * entropy coding with CAVLC or (Main and High profile) CABAC, with every context its syntax elements use, and one or more slices per picture;
//   * a choice between all of these by rate-distortion cost: squared error plus lambda times the bits the entropy coder would spend.
//
// It is not NVIDIA's encoder and does not try to produce NVIDIA's bytes. Which bitstream a given picture becomes is a pure function of the picture, the
// picture type, the QP, and (for P and B pictures) the pictures before it: an IDR picture of the same frame at the same QP is the same bytes every time.
// The reference picture a P picture predicts from is the decoded, deblocked picture, made by the decoder of this tree (h264_decode.cpp) from the bytes just
// written, so the encoder and any decoder cannot disagree about it.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nvenc_encoder.hpp"

namespace vgpu_nvenc {

struct H264Options {
  bool cabac = false;          // entropy_coding_mode_flag (Main and High profile)
  bool transform8x8 = false;   // transform_8x8_mode_flag (High profile)
  int slice_mode = 0;          // 0: slice_data macroblocks per slice, 1: at most slice_data bytes per slice, 2: slice_data macroblock rows, 3: slice_data slices
  int slice_data = 0;          // 0 with mode 0: one slice
  int num_ref = 1;             // reference pictures in list 0 of a P picture (1 .. 4)
  int max_b = 0;               // B pictures between P pictures the stream may use (0: none); the SPS tells decoders about the reordering
  bool deblock = true;
};

class H264Encoder : public VideoEncoder {
 public:
  // profile_idc: 66 (Baseline: CAVLC, no B pictures, 4x4 transform), 77 (Main) or 100 (High).
  H264Encoder(int width, int height, int fps_num, int fps_den, int profile_idc, const H264Options& opt);
  // The tools of the earlier encoder (CAVLC, 4x4 transform, one slice, one reference, no B pictures) and, for test streams, interlaced coding with field
  // pictures (frames go through encode_field_pair(); the card's NVENC refuses field encoding, so no API path reaches this). `height` is then the
  // frame's, and a multiple of 4.
  H264Encoder(int width, int height, int fps_num, int fps_den, int profile_idc, bool deblock, bool field_pictures = false);
  ~H264Encoder() override;

  std::vector<uint8_t> parameter_sets() const override;   // Annex B: SPS then PPS

  // Encodes one picture and returns its NAL units (Annex B, four-byte start codes). A P picture with no reference picture to predict from (the first, or
  // after reset()) becomes an IDR picture. `qp` is the slice QP, 0..51. `stats` may be null.
  std::vector<uint8_t> encode(const EncPicture& in, PicType type, int qp, EncStats* stats = nullptr) override;
  // With B pictures: `poc` is the picture's place in display order (the IDR picture is 0). kInter pictures are P pictures that predict from the pictures
  // before them in coding order; a kBi picture sits between two reference pictures in display order, which must both have been coded.
  std::vector<uint8_t> encode_at(const EncPicture& in, PicType type, int qp, int poc, EncStats* stats = nullptr) override;
  bool supports_b() const override;

  // Field coding only: codes a frame as two field pictures (the first an IDR, intra or P picture, the second a P picture that may
  // predict from the first). Returns both slice NAL units.
  std::vector<uint8_t> encode_field_pair(const EncPicture& frame, bool top_field_first, PicType type, int qp, EncStats* stats = nullptr);

  // Forget the reference picture and start a new coded video sequence at the next picture.
  void reset() override;

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
