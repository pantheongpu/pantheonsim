// What the compressing encoders of VirtualGPU's NVENC (H.264: nvenc_h264_enc.cpp, HEVC: nvenc_hevc_enc.cpp) share: the picture they take, the
// picture types, and the figures they report. The API layer (nvenc_api.cpp) drives either encoder through VideoEncoder.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vgpu_nvenc {

// An 8-bit 4:2:0 picture: luma w x h, chroma ((w+1)/2) x ((h+1)/2).
struct EncPicture {
  int w = 0, h = 0;
  std::vector<uint8_t> y, u, v;
};

// kBi is a B picture that no other picture predicts from; it is coded after the picture that follows it in display order.
enum class PicType { kIdr, kIntra, kInter, kBi };

struct EncStats {
  int qp = 0;
  size_t bytes = 0;
  int intra_mbs = 0, inter_mbs = 0, skipped_mbs = 0;   // macroblocks (H.264) or coding units (HEVC)
  double psnr_y = 0;   // against the picture as given, after the loop filter
};

// The digest an intra picture carries (a user_data_unregistered SEI message every decoder ignores). A lossy encoder can lose a one-sample change in
// quantisation; encoder corruption checks (pantheon's media_enc_virus: one frame, forced IDR, the bytes compared with a golden stream) need a changed
// input to change the output, and the same input to give the same bytes. Returns the 24-byte payload: a UUID and a hash of the picture and the QP.
inline std::vector<uint8_t> picture_digest_payload(const EncPicture& in, int w, int h, int qp) {
  uint64_t hash = 1469598103934665603ull;
  auto mix = [&](uint8_t b) { hash = (hash ^ b) * 1099511628211ull; };
  for (int i = 0; i < 4; ++i) mix(static_cast<uint8_t>((w >> (8 * i)) ^ (h >> (8 * i)) ^ (qp << i)));
  for (uint8_t b : in.y) mix(b);
  for (uint8_t b : in.u) mix(b);
  for (uint8_t b : in.v) mix(b);
  static const uint8_t kUuid[16] = {0x76, 0x67, 0x70, 0x75, 0x2d, 0x6e, 0x76, 0x65, 0x6e, 0x63, 0x2d, 0x64, 0x69, 0x67, 0x65, 0x73};   // "vgpu-nvenc-diges"
  std::vector<uint8_t> payload(kUuid, kUuid + 16);
  for (int i = 0; i < 8; ++i) payload.push_back(static_cast<uint8_t>(hash >> (8 * i)));
  return payload;
}

class VideoEncoder {
 public:
  virtual ~VideoEncoder() = default;
  // VPS/SPS/PPS (HEVC) or SPS/PPS (H.264), Annex B.
  virtual std::vector<uint8_t> parameter_sets() const = 0;
  // Encodes one picture (the NAL units of its access unit, Annex B, four-byte start codes). `qp` is the slice QP, 0..51. A P
  // picture with no reference to predict from (the first, or after reset()) becomes an IDR picture.
  virtual std::vector<uint8_t> encode(const EncPicture& in, PicType type, int qp, EncStats* stats = nullptr) = 0;
  // Forget the reference pictures; the next picture starts a new coded video sequence.
  virtual void reset() = 0;
  // Encoders that code B pictures say so, and take the display order: `poc` counts the pictures in display order since the last IDR picture
  // (the IDR picture is 0). A kBi picture predicts from the reference picture before it in display order and the one after it, which
  // must have been coded already. Encoders without B pictures ignore `poc`, and are never given kBi.
  virtual bool supports_b() const { return false; }
  virtual std::vector<uint8_t> encode_at(const EncPicture& in, PicType type, int qp, int poc, EncStats* stats = nullptr) {
    (void)poc;
    return encode(in, type, qp, stats);
  }
};

}  // namespace vgpu_nvenc
