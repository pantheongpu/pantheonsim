// The classic cuDNN operations whose element loops run on the host's threads (activation forward and backward, add
// and op tensor, softmax, pooling, batch normalization, and the filter gradient of depthwise and other few-chain
// convolutions), against the plain double-precision loops they replaced. Each result is compared bit for bit (two NaNs
// count as equal): the tensors are float, every element is computed in double and rounded once on the way out, so the
// loops' answers are exactly reproducible. Inputs include signed zeros, subnormals, infinities and NaN; shapes are
// large enough that the loops are cut into pieces for several threads.
// Not run against the card: this pins the library to its own arithmetic (the conformance suites pin that to the card).
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

static int failures = 0, checks = 0;
static void report(bool ok, const char* what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what);
  }
}
static bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0 || (std::isnan(a) && std::isnan(b)); }
static bool same_all(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!same_bits(a[i], b[i])) return false;
  return true;
}

static std::mt19937_64 rng(777);
enum class Mix { Plain, Special };
static std::vector<float> gen(size_t n, Mix mix, float lo = -2.f, float hi = 2.f) {
  std::uniform_real_distribution<float> u(lo, hi);
  const float sp[] = {0.f, -0.f, 1.f, -1.f, std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(),
                      std::numeric_limits<float>::min(), 1e30f, -1e30f, std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 100.f, -100.f};
  std::vector<float> v(n);
  for (auto& x : v) x = (mix == Mix::Special && rng() % 25 == 0) ? sp[rng() % (sizeof sp / sizeof sp[0])] : u(rng);
  return v;
}
static float* dev(const std::vector<float>& h) {
  float* d = nullptr;
  cudaMalloc(&d, h.size() * 4 + 16);
  cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  return d;
}
static std::vector<float> host(const float* d, size_t n) {
  std::vector<float> h(n);
  cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
  return h;
}
// Every descriptor the checks create is destroyed at the end (LeakSanitizer fails the test otherwise).
static std::vector<cudnnTensorDescriptor_t> g_tensor_descs;
static std::vector<cudnnActivationDescriptor_t> g_act_descs;
static std::vector<cudnnOpTensorDescriptor_t> g_op_descs;
static std::vector<cudnnPoolingDescriptor_t> g_pool_descs;
static std::vector<cudnnFilterDescriptor_t> g_filter_descs;
static std::vector<cudnnConvolutionDescriptor_t> g_conv_descs;
static void destroy_descriptors() {
  for (auto d : g_tensor_descs) cudnnDestroyTensorDescriptor(d);
  for (auto d : g_act_descs) cudnnDestroyActivationDescriptor(d);
  for (auto d : g_op_descs) cudnnDestroyOpTensorDescriptor(d);
  for (auto d : g_pool_descs) cudnnDestroyPoolingDescriptor(d);
  for (auto d : g_filter_descs) cudnnDestroyFilterDescriptor(d);
  for (auto d : g_conv_descs) cudnnDestroyConvolutionDescriptor(d);
}
static cudnnTensorDescriptor_t desc(int n, int c, int h, int w) {
  cudnnTensorDescriptor_t d;
  cudnnCreateTensorDescriptor(&d); g_tensor_descs.push_back(d);
  cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
  return d;
}

// out = alpha * r + beta * old, rounded to float: the library's blend.
static float blend(double r, double alpha, double beta, float old) {
  return static_cast<float>(beta == 0.0 ? alpha * r : alpha * r + beta * static_cast<double>(old));
}

static double sigm(double v) { return 1.0 / (1.0 + std::exp(-v)); }

// ---- activation ----
static void check_activation(cudnnHandle_t h) {
  const int N = 3, C = 8, H = 40, W = 40;
  const size_t n = (size_t)N * C * H * W;
  auto td = desc(N, C, H, W);
  cudnnActivationDescriptor_t ad;
  cudnnCreateActivationDescriptor(&ad); g_act_descs.push_back(ad);
  for (const Mix mix : {Mix::Plain, Mix::Special})
    for (const cudnnActivationMode_t mode : {CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU,
                                             CUDNN_ACTIVATION_ELU, CUDNN_ACTIVATION_SWISH})
      for (const float beta : {0.f, 0.5f}) {
        const double coef = 1.5, swish_beta = 0.75;
        cudnnSetActivationDescriptor(ad, mode, CUDNN_NOT_PROPAGATE_NAN, coef);
        if (mode == CUDNN_ACTIVATION_SWISH) cudnnSetActivationDescriptorSwishBeta(ad, swish_beta);
        const auto x = gen(n, mix), y0 = gen(n, Mix::Plain), dy = gen(n, mix);
        const float alpha = 1.25f;
        float *dx_ = dev(x), *dy_ = dev(y0);
        char what[160];
        std::snprintf(what, sizeof what, "ActivationForward mode %d beta %g %s", (int)mode, beta, mix == Mix::Plain ? "plain" : "special");
        cudnnActivationForward(h, ad, &alpha, td, dx_, &beta, td, dy_);
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) {
          const double v = x[i];
          double r = 0;
          switch (mode) {
            case CUDNN_ACTIVATION_SIGMOID: r = sigm(v); break;
            case CUDNN_ACTIVATION_RELU: r = v > 0.0 ? v : 0.0; break;
            case CUDNN_ACTIVATION_TANH: r = std::tanh(v); break;
            case CUDNN_ACTIVATION_CLIPPED_RELU: r = std::min(std::max(v, 0.0), coef); break;
            case CUDNN_ACTIVATION_ELU: r = v > 0.0 ? v : coef * (std::exp(v) - 1.0); break;
            default: r = v * sigm(swish_beta * v); break;
          }
          want[i] = blend(r, alpha, beta, y0[i]);
        }
        report(same_all(host(dy_, n), want), what);
        // backward: dx = alpha * f'(.) dy + beta * dx
        const bool reads_x = mode == CUDNN_ACTIVATION_RELU || mode == CUDNN_ACTIVATION_CLIPPED_RELU || mode == CUDNN_ACTIVATION_SWISH;
        const auto yv = host(dy_, n);   // the library's own y for the modes whose gradient reads y
        float *dyd = dev(dy), *ddx = dev(y0), *dy2 = dev(yv);
        cudnnActivationBackward(h, ad, &alpha, td, dy2, td, dyd, td, dx_, &beta, td, ddx);
        std::snprintf(what, sizeof what, "ActivationBackward mode %d beta %g %s", (int)mode, beta, mix == Mix::Plain ? "plain" : "special");
        for (size_t i = 0; i < n; ++i) {
          const double v = reads_x ? (double)x[i] : (double)yv[i];
          double g = dy[i];
          switch (mode) {
            case CUDNN_ACTIVATION_RELU: g = v > 0.0 ? g : 0.0; break;
            case CUDNN_ACTIVATION_CLIPPED_RELU: g = v > 0.0 && v <= coef ? g : 0.0; break;
            case CUDNN_ACTIVATION_SWISH: {
              const double s = sigm(swish_beta * v);
              g *= s + swish_beta * v * s * (1.0 - s);
              break;
            }
            case CUDNN_ACTIVATION_SIGMOID: g *= v * (1.0 - v); break;
            case CUDNN_ACTIVATION_TANH: g *= 1.0 - v * v; break;
            default: g = v >= 0.0 ? g : g * (v + coef); break;
          }
          want[i] = blend(g, alpha, beta, y0[i]);
        }
        report(same_all(host(ddx, n), want), what);
        cudaFree(dx_); cudaFree(dy_); cudaFree(dyd); cudaFree(ddx); cudaFree(dy2);
      }
}

// ---- add tensor and op tensor ----
static void check_tensor_ops(cudnnHandle_t h) {
  const int N = 3, C = 8, H = 40, W = 40;
  const size_t n = (size_t)N * C * H * W;
  auto td = desc(N, C, H, W);
  struct Shape { int n, c, h, w; };
  for (const Shape s : {Shape{1, C, 1, 1}, Shape{N, C, H, W}, Shape{1, 1, H, W}, Shape{N, 1, 1, W}})
    for (const Mix mix : {Mix::Plain, Mix::Special}) {
      auto bd = desc(s.n, s.c, s.h, s.w);
      const size_t nb = (size_t)s.n * s.c * s.h * s.w;
      const auto b = gen(nb, mix), a = gen(n, mix), c0 = gen(n, Mix::Plain);
      // broadcast index
      auto bidx = [&](size_t i) {
        const int w = i % W, hh = (i / W) % H, cc = (i / ((size_t)W * H)) % C, nn = i / ((size_t)W * H * C);
        return (size_t)(((s.n == 1 ? 0 : nn) * s.c + (s.c == 1 ? 0 : cc)) * s.h + (s.h == 1 ? 0 : hh)) * s.w + (s.w == 1 ? 0 : w);
      };
      char what[160];
      for (const float beta : {1.f, 0.f}) {
        const float alpha = 0.5f;
        float *db = dev(b), *dc = dev(c0);
        cudnnAddTensor(h, &alpha, bd, db, &beta, td, dc);
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = blend(b[bidx(i)], alpha, beta, c0[i]);
        std::snprintf(what, sizeof what, "AddTensor B [%d,%d,%d,%d] beta %g %s", s.n, s.c, s.h, s.w, beta, mix == Mix::Plain ? "plain" : "special");
        report(same_all(host(dc, n), want), what);
        cudaFree(db); cudaFree(dc);
      }
      cudnnOpTensorDescriptor_t od;
      cudnnCreateOpTensorDescriptor(&od); g_op_descs.push_back(od);
      for (const cudnnOpTensorOp_t op : {CUDNN_OP_TENSOR_ADD, CUDNN_OP_TENSOR_MUL, CUDNN_OP_TENSOR_MIN, CUDNN_OP_TENSOR_MAX, CUDNN_OP_TENSOR_SQRT,
                                         CUDNN_OP_TENSOR_NOT}) {
        cudnnSetOpTensorDescriptor(od, op, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
        const float a1 = 0.75f, a2 = -1.5f, beta = 0.25f;
        float *da = dev(a), *db = dev(b), *dc = dev(c0);
        cudnnOpTensor(h, od, &a1, td, da, &a2, bd, db, &beta, td, dc);
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) {
          const double av = (double)a1 * a[i], bv = (double)a2 * b[bidx(i)];
          double r = 0;
          switch (op) {
            case CUDNN_OP_TENSOR_ADD: r = av + bv; break;
            case CUDNN_OP_TENSOR_MUL: r = av * bv; break;
            case CUDNN_OP_TENSOR_MIN: r = std::min(av, bv); break;
            case CUDNN_OP_TENSOR_MAX: r = std::max(av, bv); break;
            case CUDNN_OP_TENSOR_SQRT: r = std::sqrt(av); break;
            default: r = 1.0 - av; break;
          }
          want[i] = blend(r, 1.0, beta, c0[i]);
        }
        std::snprintf(what, sizeof what, "OpTensor %d B [%d,%d,%d,%d] %s", (int)op, s.n, s.c, s.h, s.w, mix == Mix::Plain ? "plain" : "special");
        report(same_all(host(dc, n), want), what);
        cudaFree(da); cudaFree(db); cudaFree(dc);
      }
    }
}

// ---- softmax ----
static void check_softmax(cudnnHandle_t h) {
  const int N = 3, C = 10, H = 12, W = 12;
  const size_t n = (size_t)N * C * H * W;
  auto td = desc(N, C, H, W);
  for (const Mix mix : {Mix::Plain, Mix::Special})
    for (const cudnnSoftmaxMode_t mode : {CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL})
      for (const cudnnSoftmaxAlgorithm_t algo : {CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG}) {
        const auto x = gen(n, mix, -3.f, 3.f), y0 = gen(n, Mix::Plain), dy = gen(n, mix);
        const float alpha = 1.f, beta = 0.f;
        // groups: INSTANCE over C*H*W per n; CHANNEL over C per (n, h, w)
        const size_t S = (size_t)H * W;
        auto group = [&](size_t g, std::vector<size_t>* idx) {
          idx->clear();
          if (mode == CUDNN_SOFTMAX_MODE_INSTANCE) for (size_t i = 0; i < (size_t)C * S; ++i) idx->push_back(g * C * S + i);
          else for (size_t c = 0; c < (size_t)C; ++c) idx->push_back(((g / S) * C + c) * S + g % S);
        };
        const size_t ngroups = mode == CUDNN_SOFTMAX_MODE_INSTANCE ? (size_t)N : (size_t)N * S;
        float *dx_ = dev(x), *dy_ = dev(y0);
        cudnnSoftmaxForward(h, algo, mode, &alpha, td, dx_, &beta, td, dy_);
        std::vector<float> want(n);
        std::vector<size_t> idx;
        for (size_t g = 0; g < ngroups; ++g) {
          group(g, &idx);
          double mx = -INFINITY, sum = 0.0;
          for (size_t i : idx) mx = std::max(mx, (double)x[i]);
          for (size_t i : idx) sum += std::exp(x[i] - mx);
          for (size_t i : idx) want[i] = blend(algo == CUDNN_SOFTMAX_LOG ? x[i] - mx - std::log(sum) : std::exp(x[i] - mx) / sum, alpha, beta, y0[i]);
        }
        char what[160];
        std::snprintf(what, sizeof what, "SoftmaxForward mode %d algo %d %s", (int)mode, (int)algo, mix == Mix::Plain ? "plain" : "special");
        report(same_all(host(dy_, n), want), what);
        const auto yv = host(dy_, n);
        float *dyd = dev(dy), *ddx = dev(y0), *dy2 = dev(yv);
        cudnnSoftmaxBackward(h, algo, mode, &alpha, td, dy2, td, dyd, &beta, td, ddx);
        for (size_t g = 0; g < ngroups; ++g) {
          group(g, &idx);
          double sum = 0.0;
          if (algo == CUDNN_SOFTMAX_LOG) {
            for (size_t i : idx) sum += dy[i];
            for (size_t i : idx) want[i] = blend(dy[i] - std::exp((double)yv[i]) * sum, alpha, beta, y0[i]);
          } else {
            for (size_t i : idx) sum += (double)yv[i] * dy[i];
            for (size_t i : idx) want[i] = blend(yv[i] * (dy[i] - sum), alpha, beta, y0[i]);
          }
        }
        std::snprintf(what, sizeof what, "SoftmaxBackward mode %d algo %d %s", (int)mode, (int)algo, mix == Mix::Plain ? "plain" : "special");
        report(same_all(host(ddx, n), want), what);
        cudaFree(dx_); cudaFree(dy_); cudaFree(dyd); cudaFree(ddx); cudaFree(dy2);
      }
}

// ---- pooling ----
static void check_pooling(cudnnHandle_t h) {
  const int N = 2, C = 6, H = 31, W = 29;
  struct Cfg { int win, pad, stride; };
  for (const Cfg cfg : {Cfg{2, 0, 2}, Cfg{3, 1, 2}, Cfg{3, 1, 1}})
    for (const cudnnPoolingMode_t mode : {CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING})
      for (const Mix mix : {Mix::Plain, Mix::Special}) {
        const int OH = 1 + (H + 2 * cfg.pad - cfg.win) / cfg.stride, OW = 1 + (W + 2 * cfg.pad - cfg.win) / cfg.stride;
        cudnnPoolingDescriptor_t pd;
        cudnnCreatePoolingDescriptor(&pd); g_pool_descs.push_back(pd);
        cudnnSetPooling2dDescriptor(pd, mode, CUDNN_NOT_PROPAGATE_NAN, cfg.win, cfg.win, cfg.pad, cfg.pad, cfg.stride, cfg.stride);
        auto xd = desc(N, C, H, W), yd = desc(N, C, OH, OW);
        const size_t nx = (size_t)N * C * H * W, ny = (size_t)N * C * OH * OW;
        const auto x = gen(nx, mix), y0 = gen(ny, Mix::Plain), dy = gen(ny, mix), dx0 = gen(nx, Mix::Plain);
        const float alpha = 1.f, beta = 0.5f;
        float *dxp = dev(x), *dyp = dev(y0);
        cudnnPoolingForward(h, pd, &alpha, xd, dxp, &beta, yd, dyp);
        auto window = [&](int o, auto&& f) {  // f(input offset within the plane)
          const int oy = o / OW, ox = o % OW;
          for (int i = 0; i < cfg.win; ++i) {
            const int py = oy * cfg.stride - cfg.pad + i;
            if (py < 0 || py >= H) continue;
            for (int j = 0; j < cfg.win; ++j) {
              const int px = ox * cfg.stride - cfg.pad + j;
              if (px < 0 || px >= W) continue;
              f(py * W + px);
            }
          }
        };
        std::vector<float> want(ny);
        for (int nc = 0; nc < N * C; ++nc)
          for (int o = 0; o < OH * OW; ++o) {
            const float* plane = x.data() + (size_t)nc * H * W;
            double best = -INFINITY, sum = 0.0;
            int64_t count = 0;
            window(o, [&](int at) { best = std::max(best, (double)plane[at]); sum += plane[at]; ++count; });
            double v = mode == CUDNN_POOLING_MAX ? best
                       : mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING ? sum / (double)(cfg.win * cfg.win)
                                                                             : (count ? sum / (double)count : 0.0);
            want[(size_t)nc * OH * OW + o] = blend(v, alpha, beta, y0[(size_t)nc * OH * OW + o]);
          }
        char what[160];
        std::snprintf(what, sizeof what, "PoolingForward mode %d win %d pad %d stride %d %s", (int)mode, cfg.win, cfg.pad, cfg.stride, mix == Mix::Plain ? "plain" : "special");
        report(same_all(host(dyp, ny), want), what);
        const auto yv = host(dyp, ny);
        float *ddy = dev(dy), *ddx = dev(dx0), *dyv = dev(yv);
        cudnnPoolingBackward(h, pd, &alpha, yd, dyv, yd, ddy, xd, dxp, &beta, xd, ddx);
        std::vector<double> r(nx, 0.0);
        std::vector<float> want2(nx);
        for (int nc = 0; nc < N * C; ++nc)
          for (int o = 0; o < OH * OW; ++o) {
            double* out = r.data() + (size_t)nc * H * W;
            const double grad = dy[(size_t)nc * OH * OW + o];
            if (mode == CUDNN_POOLING_MAX) {
              const float* plane = x.data() + (size_t)nc * H * W;
              int arg = -1;
              double best = 0.0;
              window(o, [&](int at) { if (arg < 0 || plane[at] > best) arg = at, best = plane[at]; });
              if (arg >= 0) out[arg] += grad;
            } else {
              int64_t count = 0;
              window(o, [&](int) { ++count; });
              const double div = mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING ? (double)(cfg.win * cfg.win) : (double)count;
              if (count) window(o, [&](int at) { out[at] += grad / div; });
            }
          }
        for (size_t i = 0; i < nx; ++i) want2[i] = blend(r[i], alpha, beta, dx0[i]);
        std::snprintf(what, sizeof what, "PoolingBackward mode %d win %d pad %d stride %d %s", (int)mode, cfg.win, cfg.pad, cfg.stride, mix == Mix::Plain ? "plain" : "special");
        report(same_all(host(ddx, nx), want2), what);
        cudaFree(dxp); cudaFree(dyp); cudaFree(ddy); cudaFree(ddx); cudaFree(dyv);
      }
}

// ---- batch normalization ----
static void check_batchnorm(cudnnHandle_t h) {
  const int N = 3, C = 6, H = 14, W = 14;
  const size_t n = (size_t)N * C * H * W, S = (size_t)H * W;
  auto xd = desc(N, C, H, W);
  for (const cudnnBatchNormMode_t mode : {CUDNN_BATCHNORM_SPATIAL, CUDNN_BATCHNORM_PER_ACTIVATION})
    for (const Mix mix : {Mix::Plain, Mix::Special}) {
      const bool spatial = mode == CUDNN_BATCHNORM_SPATIAL;
      const int pc = C, ph = spatial ? 1 : H, pw = spatial ? 1 : W;
      auto bd = desc(1, pc, ph, pw);
      const size_t np = (size_t)pc * ph * pw;
      auto slot = [&](size_t c, size_t i) { return spatial ? c : c * S + i; };
      const auto x = gen(n, mix), y0 = gen(n, Mix::Plain), scale = gen(np, Mix::Plain, 0.5f, 1.5f), bias = gen(np, Mix::Plain), rm0 = gen(np, Mix::Plain), rv0 = gen(np, Mix::Plain, 0.5f, 1.5f), dy = gen(n, mix);
      const float one = 1.f, zero = 0.f;
      const double eps = 1e-5;
      float *dx_ = dev(x), *dyo = dev(y0), *dsc = dev(scale), *dbi = dev(bias), *drm = dev(rm0), *drv = dev(rv0);
      std::vector<float> sm0(np, 0.f), si0(np, 0.f);
      float *dsm = dev(sm0), *dsi = dev(si0);
      cudnnBatchNormalizationForwardTraining(h, mode, &one, &zero, xd, dx_, xd, dyo, bd, dsc, dbi, 0.1, drm, drv, eps, dsm, dsi);
      // reference
      std::vector<double> sum(np, 0.0), sq(np, 0.0), cnt(np, 0.0);
      for (size_t nn = 0; nn < (size_t)N; ++nn)
        for (size_t c = 0; c < (size_t)C; ++c)
          for (size_t i = 0; i < S; ++i) {
            const size_t p = slot(c, i);
            const double v = x[(nn * C + c) * S + i];
            sum[p] += v, sq[p] += v * v, cnt[p] += 1;
          }
      std::vector<double> mean(np), var(np), inv(np);
      for (size_t p = 0; p < np; ++p) {
        mean[p] = cnt[p] ? sum[p] / cnt[p] : 0.0;
        var[p] = cnt[p] ? std::max(0.0, sq[p] / cnt[p] - mean[p] * mean[p]) : 0.0;
        inv[p] = 1.0 / std::sqrt(var[p] + eps);
      }
      std::vector<float> want(n);
      for (size_t nn = 0; nn < (size_t)N; ++nn)
        for (size_t c = 0; c < (size_t)C; ++c)
          for (size_t i = 0; i < S; ++i) {
            const size_t p = slot(c, i), at = (nn * C + c) * S + i;
            want[at] = blend(scale[p] * (x[at] - mean[p]) * inv[p] + bias[p], 1.0, 0.0, y0[at]);
          }
      char what[160];
      std::snprintf(what, sizeof what, "BatchNormalizationForwardTraining %s %s", spatial ? "spatial" : "per-activation", mix == Mix::Plain ? "plain" : "special");
      report(same_all(host(dyo, n), want), what);
      std::vector<float> wsm(np), wsi(np), wrm(np), wrv(np);
      for (size_t p = 0; p < np; ++p) {
        wsm[p] = (float)mean[p];
        wsi[p] = (float)inv[p];
        const double unbiased = cnt[p] > 1 ? var[p] * cnt[p] / (cnt[p] - 1) : var[p];
        wrm[p] = (float)(0.1 * mean[p] + (1.0 - 0.1) * rm0[p]);
        wrv[p] = (float)(0.1 * unbiased + (1.0 - 0.1) * rv0[p]);
      }
      std::snprintf(what, sizeof what, "BatchNormalizationForwardTraining statistics %s %s", spatial ? "spatial" : "per-activation", mix == Mix::Plain ? "plain" : "special");
      report(same_all(host(dsm, np), wsm) && same_all(host(dsi, np), wsi) && same_all(host(drm, np), wrm) && same_all(host(drv, np), wrv), what);
      // inference
      float* dyi = dev(y0);
      cudnnBatchNormalizationForwardInference(h, mode, &one, &zero, xd, dx_, xd, dyi, bd, dsc, dbi, drm, drv, eps);
      const auto rm = host(drm, np), rv = host(drv, np);
      for (size_t nn = 0; nn < (size_t)N; ++nn)
        for (size_t c = 0; c < (size_t)C; ++c)
          for (size_t i = 0; i < S; ++i) {
            const size_t p = slot(c, i), at = (nn * C + c) * S + i;
            want[at] = blend(scale[p] * (x[at] - (double)rm[p]) / std::sqrt(eps + (double)rv[p]) + bias[p], 1.0, 0.0, y0[at]);
          }
      std::snprintf(what, sizeof what, "BatchNormalizationForwardInference %s %s", spatial ? "spatial" : "per-activation", mix == Mix::Plain ? "plain" : "special");
      report(same_all(host(dyi, n), want), what);
      // backward, with the saved statistics
      float *ddy = dev(dy), *ddx = dev(y0), *dds = dev(std::vector<float>(np, 0.f)), *ddb = dev(std::vector<float>(np, 0.f));
      cudnnBatchNormalizationBackward(h, mode, &one, &zero, &one, &zero, xd, dx_, xd, ddy, xd, ddx, bd, dsc, dds, ddb, eps, dsm, dsi);
      std::vector<double> dbias(np, 0.0), dscale(np, 0.0);
      const auto mean_f = host(dsm, np), inv_f = host(dsi, np);
      for (size_t nn = 0; nn < (size_t)N; ++nn)
        for (size_t c = 0; c < (size_t)C; ++c)
          for (size_t i = 0; i < S; ++i) {
            const size_t p = slot(c, i), at = (nn * C + c) * S + i;
            dbias[p] += dy[at];
            dscale[p] += dy[at] * (x[at] - (double)mean_f[p]) * (double)inv_f[p];
          }
      for (size_t nn = 0; nn < (size_t)N; ++nn)
        for (size_t c = 0; c < (size_t)C; ++c)
          for (size_t i = 0; i < S; ++i) {
            const size_t p = slot(c, i), at = (nn * C + c) * S + i;
            const double xhat = (x[at] - (double)mean_f[p]) * (double)inv_f[p];
            want[at] = blend((double)scale[p] * inv_f[p] / cnt[p] * (cnt[p] * dy[at] - dbias[p] - xhat * dscale[p]), 1.0, 0.0, y0[at]);
          }
      std::vector<float> wds(np), wdb(np);
      for (size_t p = 0; p < np; ++p) wds[p] = (float)dscale[p], wdb[p] = (float)dbias[p];
      std::snprintf(what, sizeof what, "BatchNormalizationBackward %s %s", spatial ? "spatial" : "per-activation", mix == Mix::Plain ? "plain" : "special");
      report(same_all(host(ddx, n), want) && same_all(host(dds, np), wds) && same_all(host(ddb, np), wdb), what);
      cudaFree(dx_); cudaFree(dyo); cudaFree(dsc); cudaFree(dbi); cudaFree(drm); cudaFree(drv); cudaFree(dsm); cudaFree(dsi);
      cudaFree(dyi); cudaFree(ddy); cudaFree(ddx); cudaFree(dds); cudaFree(ddb);
    }
}

// ---- depthwise (and other few-chain) filter gradients ----
static void check_conv_filter(cudnnHandle_t h) {
  struct Cfg { int N, C, K, groups, H, W, R, S, pad, stride, dil; };
  for (const Cfg c : {Cfg{2, 8, 8, 8, 17, 19, 3, 3, 1, 1, 1}, Cfg{2, 6, 6, 6, 16, 16, 5, 5, 2, 1, 1}, Cfg{3, 4, 4, 4, 15, 21, 3, 7, 1, 2, 1},
                      Cfg{2, 8, 16, 4, 12, 12, 3, 3, 1, 1, 1}, Cfg{2, 4, 4, 4, 9, 11, 3, 3, 2, 1, 2}, Cfg{1, 6, 6, 6, 3, 4, 3, 3, 2, 1, 1}})
    for (const Mix mix : {Mix::Plain, Mix::Special}) {
      const int Cg = c.C / c.groups, Kg = c.K / c.groups;
      const int OH = 1 + (c.H + 2 * c.pad - ((c.R - 1) * c.dil + 1)) / c.stride, OW = 1 + (c.W + 2 * c.pad - ((c.S - 1) * c.dil + 1)) / c.stride;
      cudnnFilterDescriptor_t fd;
      cudnnCreateFilterDescriptor(&fd); g_filter_descs.push_back(fd);
      cudnnSetFilter4dDescriptor(fd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, c.K, Cg, c.R, c.S);
      cudnnConvolutionDescriptor_t cd;
      cudnnCreateConvolutionDescriptor(&cd); g_conv_descs.push_back(cd);
      cudnnSetConvolution2dDescriptor(cd, c.pad, c.pad, c.stride, c.stride, c.dil, c.dil, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
      cudnnSetConvolutionGroupCount(cd, c.groups);
      auto xd = desc(c.N, c.C, c.H, c.W), yd = desc(c.N, c.K, OH, OW);
      const size_t nx = (size_t)c.N * c.C * c.H * c.W, ny = (size_t)c.N * c.K * OH * OW, nw = (size_t)c.K * Cg * c.R * c.S;
      const auto x = gen(nx, mix), dy = gen(ny, mix), w0 = gen(nw, Mix::Plain);
      float *dxp = dev(x), *dyp = dev(dy), *dwp = dev(w0);
      void* ws = nullptr;
      cudaMalloc(&ws, 1 << 26);
      const float one = 1.f, zero = 0.f;
      const cudnnStatus_t st = cudnnConvolutionBackwardFilter(h, &one, xd, dxp, yd, dyp, cd, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, ws, 1 << 26, &zero, fd, dwp);
      std::vector<float> want(nw);
      for (int k = 0; k < c.K; ++k) {
        const int c0 = (k / Kg) * Cg;
        for (int ci = 0; ci < Cg; ++ci)
          for (int f0 = 0; f0 < c.R; ++f0)
            for (int f1 = 0; f1 < c.S; ++f1) {
              double s = 0.0;
              for (int n = 0; n < c.N; ++n)
                for (int o0 = 0; o0 < OH; ++o0) {
                  const int p0 = o0 * c.stride - c.pad + f0 * c.dil;
                  if (p0 < 0 || p0 >= c.H) continue;
                  for (int o1 = 0; o1 < OW; ++o1) {
                    const int p1 = o1 * c.stride - c.pad + f1 * c.dil;
                    if (p1 < 0 || p1 >= c.W) continue;
                    s = s + (double)dy[(((size_t)n * c.K + k) * OH + o0) * OW + o1] * (double)x[(((size_t)n * c.C + c0 + ci) * c.H + p0) * c.W + p1];
                  }
                }
              want[(((size_t)k * Cg + ci) * c.R + f0) * c.S + f1] = (float)s;
            }
      }
      char what[160];
      std::snprintf(what, sizeof what, "ConvolutionBackwardFilter N%d C%d K%d groups %d %dx%d filter %dx%d pad %d stride %d dil %d %s", c.N, c.C, c.K, c.groups, c.H, c.W,
                    c.R, c.S, c.pad, c.stride, c.dil, mix == Mix::Plain ? "plain" : "special");
      report(st == CUDNN_STATUS_SUCCESS && same_all(host(dwp, nw), want), what);
      cudaFree(dxp); cudaFree(dyp); cudaFree(dwp); cudaFree(ws);
    }
}

int main() {
  cudnnHandle_t h;
  if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) {
    std::printf("SKIP: cudnnCreate failed\n");
    return 0;
  }
  check_activation(h);
  check_tensor_ops(h);
  check_softmax(h);
  check_pooling(h);
  check_batchnorm(h);
  check_conv_filter(h);
  destroy_descriptors();
  cudnnDestroy(h);
  std::printf("%d checks, %d failed\n", checks, failures);
  std::printf(failures ? "FAIL\n" : "PASS\n");
  return failures ? 1 : 0;
}
