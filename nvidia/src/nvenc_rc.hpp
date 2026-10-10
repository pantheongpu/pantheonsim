// Rate control for VirtualGPU's NVENC: picks the QP of each picture so that a stream's average bit rate lands on the application's target.
//
// The model. A picture of type T (intra, P, B) at quantiser step Q takes about X_T * Q^-alpha bits, X_T being a complexity the controller learns from the
// pictures it has seen (Q = 2^((QP - 4) / 6); alpha = 0.85 until two trial encodes of one picture say otherwise). The pattern of the stream -- how many pictures
// are intra, how many B -- and the QP offsets between the types give the expected bits per picture at a common QP; the QP at which that equals the bits the
// stream can still afford per picture is the working point, and each type's target size follows. The affordable bits per picture are the nominal ones plus a
// share (1/window) of what the stream is behind or ahead of its budget, so an overshoot is paid back within a few pictures (CBR: 8, VBR: 24).
//
// The caller encodes a picture at qp_for(), compares the size with target_bits(), and may encode it again at a corrected QP (next_qp()) after the encoder
// rolled the first attempt back; it reports the final size with update(). A video buffering verifier (a leaky bucket of vbv_bits, drained by the pictures and
// filled at the bit rate; for VBR at the peak rate) bounds the size of one picture so that the buffer never underflows.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vgpu_nvenc {

class RateController {
 public:
  enum Type { kIntra = 0, kP = 1, kB = 2 };

  struct Config {
    double bitrate = 0;       // average bits per second; 0: rate control is off
    double max_bitrate = 0;   // peak bits per second (VBR); 0: the average
    double fps = 30;
    bool cbr = false;
    double vbv_bits = 0;      // buffer size; 0: one second of the (peak) rate
    double intra_fraction = 0;   // pictures that are intra: 1 / GOP length, 1 for an intra-only stream
    int b_per_p = 0;          // B pictures per P picture
    int b_offset = 2;         // QP of a B picture above a P picture's
    int qp_min[3] = {0, 0, 0}, qp_max[3] = {51, 51, 51};
  };

  void configure(const Config& c) {
    cfg_ = c;
    const double rate = cfg_.max_bitrate > 0 ? cfg_.max_bitrate : cfg_.bitrate;
    vbv_ = cfg_.vbv_bits > 0 ? cfg_.vbv_bits : rate;
    fullness_ = 0.9 * vbv_;
    total_target_ = total_spent_ = 0;
    for (int t = 0; t < 3; ++t) last_qp_[t] = -1;
  }
  bool active() const { return cfg_.bitrate > 0; }
  const Config& config() const { return cfg_; }

  // bits an average picture may take: the nominal bits, corrected by the budget the stream is ahead of or behind
  double affordable() const {
    const double nominal = cfg_.bitrate / cfg_.fps;
    const double window = cfg_.cbr ? 8.0 : 24.0;
    const double correction = (total_target_ - total_spent_) / window;
    return std::min(std::max(nominal + correction, 0.4 * nominal), 1.8 * nominal);
  }

  // The QP of a picture of the type, as a real number: the working point plus the type's offset.
  double qp_real(Type t) const {
    const double q = working_point();
    return std::min<double>(cfg_.qp_max[t], std::max<double>(cfg_.qp_min[t], q + offset(t)));
  }
  // The QP to code a picture of the type at: the working point's, moved by at most three steps from the QP of the last picture of the type.
  int qp_for(Type t) const {
    int q = static_cast<int>(std::lround(qp_real(t)));
    if (last_qp_[t] >= 0) q = std::min(last_qp_[t] + 3, std::max(last_qp_[t] - 3, q));
    return std::min(cfg_.qp_max[t], std::max(cfg_.qp_min[t], q));
  }

  // Bits a picture of the type should take at the working point.
  double target_bits(Type t) const { return bits_at(t, working_point() + offset(t)); }

  // The QP to try after a picture of `bits` at `qp` missed its target: the one the model says hits it (at least one step in the right direction).
  int next_qp(Type t, int qp, double bits, double target) const {
    const double want = qp + 6.0 / alpha_ * std::log2(bits / std::max(target, 1.0));
    int q = static_cast<int>(std::lround(want));
    q = std::min(qp + 5, std::max(qp - 5, q));
    if (q == qp) q += bits > target ? 1 : -1;
    return std::min(cfg_.qp_max[t], std::max(cfg_.qp_min[t], q));
  }
  // Largest size a picture may take without underflowing the buffer.
  double max_bits() const { return fullness_ + arrival() - 0.02 * vbv_; }

  // Records a coded picture.
  void update(Type t, int qp, double bits) {
    const double x = bits * std::pow(qstep(qp), alpha_);
    x_[t] = have_[t] ? 0.5 * x_[t] + 0.5 * x : x;
    have_[t] = true;
    last_qp_[t] = qp;
    total_target_ += cfg_.bitrate / cfg_.fps;
    total_spent_ += bits;
    fullness_ = std::min(vbv_, std::max(0.0, fullness_ + arrival() - bits));
  }
  // Two attempts at one picture: learn the exponent of the model.
  void learn_alpha(int qp_a, double bits_a, int qp_b, double bits_b) {
    if (qp_a == qp_b || bits_a < 64 || bits_b < 64) return;
    const double a = (std::log2(bits_a) - std::log2(bits_b)) / ((qp_b - qp_a) / 6.0);
    alpha_ = std::min(1.2, std::max(0.45, 0.7 * alpha_ + 0.3 * a));
  }
  // The size an intra picture was given before any P picture exists, to scale the other types from.
  bool known(Type t) const { return have_[t]; }
  double spent() const { return total_spent_; }
  double budget() const { return total_target_; }
  double alpha() const { return alpha_; }

 private:
  static double qstep(double qp) { return std::exp2((qp - 4.0) / 6.0); }
  double arrival() const {
    const double rate = cfg_.cbr || cfg_.max_bitrate <= 0 ? cfg_.bitrate : cfg_.max_bitrate;
    return rate / cfg_.fps;
  }
  double offset(Type t) const { return t == kIntra ? -1.0 : (t == kB ? cfg_.b_offset : 0.0); }
  double x_of(Type t) const {
    if (have_[t]) return x_[t];
    // before a type has been seen: derived from the types that have (a P picture takes about a third of an intra picture's bits at one QP, a B picture about
    // two thirds of a P picture's); with none seen, a guess from the picture rate
    const double gap_ip = 0.33, gap_pb = 0.55;
    if (t == kIntra) return have_[kP] ? x_[kP] / gap_ip : (have_[kB] ? x_[kB] / (gap_ip * gap_pb) : prior());
    if (t == kP) return have_[kIntra] ? x_[kIntra] * gap_ip : (have_[kB] ? x_[kB] / gap_pb : prior() * gap_ip);
    return have_[kP] ? x_[kP] * gap_pb : (have_[kIntra] ? x_[kIntra] * gap_ip * gap_pb : prior() * gap_ip * gap_pb);
  }
  double prior() const { return cfg_.bitrate / cfg_.fps * std::pow(qstep(28), alpha_) * 3.0; }
  double bits_at(Type t, double qp) const { return x_of(t) * std::pow(qstep(qp), -alpha_); }
  // the QP at which the expected bits per picture equal what the stream can afford
  double working_point() const {
    const double afford = affordable();
    const double fi = cfg_.intra_fraction;
    const double fb = (1.0 - fi) * cfg_.b_per_p / (cfg_.b_per_p + 1.0);
    const double fp = 1.0 - fi - fb;
    double lo = -12.0, hi = 60.0;   // expected bits fall as the QP rises
    for (int i = 0; i < 40; ++i) {
      const double q = 0.5 * (lo + hi);
      const double e = fi * bits_at(kIntra, q + offset(kIntra)) + fp * bits_at(kP, q) + fb * bits_at(kB, q + offset(kB));
      if (e > afford) lo = q;
      else hi = q;
    }
    return 0.5 * (lo + hi);
  }

  Config cfg_;
  double x_[3] = {0, 0, 0};
  bool have_[3] = {false, false, false};
  int last_qp_[3] = {-1, -1, -1};
  double alpha_ = 0.85;
  double total_target_ = 0, total_spent_ = 0;
  double fullness_ = 0, vbv_ = 0;
};

}  // namespace vgpu_nvenc
