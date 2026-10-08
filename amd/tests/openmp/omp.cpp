// OpenMP target offloading, built by ROCm's amdclang++ (-fopenmp --offload-arch=...)
// and run on simulated AMD GPUs. The runtime is LLVM's libomptarget, whose AMD
// plugin talks to the HSA runtime directly, so this reaches the simulator's HSA
// layer (queues, signals, memory pools, executables) the way no HIP program does.
//
// Every check prints "ok" or "FAIL"; the program exits non-zero on any failure.
#include <omp.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failed = 0;
static int g_checks = 0;
#define CHECK(cond, ...)                                  \
  do {                                                    \
    ++g_checks;                                           \
    if (!(cond)) {                                        \
      ++g_failed;                                         \
      std::printf("FAIL line %d: %s: ", __LINE__, #cond); \
      std::printf(__VA_ARGS__);                           \
      std::printf("\n");                                  \
    }                                                     \
  } while (0)

#pragma omp declare target
int g_device_counter = 5;
double g_table[8] = {1, 2, 3, 4, 5, 6, 7, 8};
static double scale_by_table(int i) { return g_table[i & 7] * 2.0; }
#pragma omp end declare target

static void devices() {
  const int n = omp_get_num_devices();
  std::printf("devices %d default %d initial %d\n", n, omp_get_default_device(), omp_get_initial_device());
  CHECK(n >= 1, "no offload device (%d)", n);
  int on_host = 1;
#pragma omp target map(from : on_host)
  { on_host = omp_is_initial_device(); }
  CHECK(on_host == 0, "the target region ran on the host");
}

static void saxpy() {
  const int n = 100003;
  std::vector<float> x(n), y(n), want(n);
  for (int i = 0; i < n; ++i) {
    x[i] = 0.5f * static_cast<float>(i % 97);
    y[i] = static_cast<float>(i % 13);
    want[i] = 2.5f * x[i] + y[i];
  }
  float* xp = x.data();
  float* yp = y.data();
#pragma omp target teams distribute parallel for map(to : xp[0 : n]) map(tofrom : yp[0 : n])
  for (int i = 0; i < n; ++i) yp[i] = 2.5f * xp[i] + yp[i];
  int wrong = 0;
  for (int i = 0; i < n; ++i) wrong += y[i] != want[i];
  CHECK(wrong == 0, "saxpy: %d of %d wrong", wrong, n);
}

static void reductions() {
  const int n = 50000;
  std::vector<double> v(n);
  for (int i = 0; i < n; ++i) v[i] = (i % 1000) - 400.0;
  double* vp = v.data();
  double sum = 0, mx = -1e300, mn = 1e300;
  long count = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum, count) reduction(max : mx) reduction(min : mn) \
    map(to : vp[0 : n])
  for (int i = 0; i < n; ++i) {
    sum += vp[i];
    mx = vp[i] > mx ? vp[i] : mx;
    mn = vp[i] < mn ? vp[i] : mn;
    count += vp[i] > 0;
  }
  double hs = 0, hmx = -1e300, hmn = 1e300;
  long hc = 0;
  for (int i = 0; i < n; ++i) {
    hs += v[i];
    hmx = std::fmax(hmx, v[i]);
    hmn = std::fmin(hmn, v[i]);
    hc += v[i] > 0;
  }
  CHECK(sum == hs, "sum %.1f != %.1f", sum, hs);
  CHECK(mx == hmx && mn == hmn, "max %.1f/%.1f min %.1f/%.1f", mx, hmx, mn, hmn);
  CHECK(count == hc, "count %ld != %ld", count, hc);
}

static void data_regions() {
  const int n = 4096;
  std::vector<int> a(n), b(n, 0);
  for (int i = 0; i < n; ++i) a[i] = i;
  int* ap = a.data();
  int* bp = b.data();
#pragma omp target enter data map(to : ap[0 : n]) map(alloc : bp[0 : n])
#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; ++i) bp[i] = ap[i] * 3;
  // The device copy is ahead of the host's until the update.
  CHECK(b[10] == 0, "the host copy changed before the update");
#pragma omp target update from(bp[0 : n])
  int wrong = 0;
  for (int i = 0; i < n; ++i) wrong += b[i] != i * 3;
  CHECK(wrong == 0, "target update from: %d wrong", wrong);
  for (int i = 0; i < n; ++i) a[i] = 7;
#pragma omp target update to(ap[0 : n])
#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; ++i) bp[i] = ap[i] + 1;
#pragma omp target exit data map(from : bp[0 : n]) map(delete : ap[0 : n])
  wrong = 0;
  for (int i = 0; i < n; ++i) wrong += b[i] != 8;
  CHECK(wrong == 0, "update to, then exit data from: %d wrong", wrong);
}

static void declare_target() {
  int base = 0;
#pragma omp target map(from : base)
  { base = g_device_counter; }
  CHECK(base == 5, "declare target scalar %d", base);
#pragma omp target
  { g_device_counter += 10; }
#pragma omp target update from(g_device_counter)
  CHECK(g_device_counter == 15, "declare target scalar after update %d", g_device_counter);
  const int n = 64;
  std::vector<double> out(n);
  double* op = out.data();
#pragma omp target teams distribute parallel for map(from : op[0 : n])
  for (int i = 0; i < n; ++i) op[i] = scale_by_table(i);
  int wrong = 0;
  for (int i = 0; i < n; ++i) wrong += out[i] != (static_cast<double>(i & 7) + 1) * 2.0;
  CHECK(wrong == 0, "declare target function and table: %d wrong", wrong);
}

static void device_memory() {
  const int dev = omp_get_default_device();
  const size_t n = 1000;
  int* d = static_cast<int*>(omp_target_alloc(n * sizeof(int), dev));
  CHECK(d != nullptr, "omp_target_alloc");
  if (!d) return;
  std::vector<int> h(n), back(n, -1);
  for (size_t i = 0; i < n; ++i) h[i] = static_cast<int>(i * i);
  CHECK(omp_target_memcpy(d, h.data(), n * sizeof(int), 0, 0, dev, omp_get_initial_device()) == 0, "memcpy to");
#pragma omp target teams distribute parallel for is_device_ptr(d)
  for (size_t i = 0; i < n; ++i) d[i] += 1;
  CHECK(omp_target_memcpy(back.data(), d, n * sizeof(int), 0, 0, omp_get_initial_device(), dev) == 0, "memcpy from");
  int wrong = 0;
  for (size_t i = 0; i < n; ++i) wrong += back[i] != static_cast<int>(i * i) + 1;
  CHECK(wrong == 0, "device memory round trip: %d wrong", wrong);
  // A host buffer made present on the device by hand.
  int* d2 = static_cast<int*>(omp_target_alloc(n * sizeof(int), dev));
  CHECK(omp_target_associate_ptr(h.data(), d2, n * sizeof(int), 0, dev) == 0, "associate");
  CHECK(omp_target_is_present(h.data(), dev) != 0, "present after associate");
  CHECK(omp_target_disassociate_ptr(h.data(), dev) == 0, "disassociate");
  CHECK(omp_target_is_present(h.data(), dev) == 0, "absent after disassociate");
  omp_target_free(d2, dev);
  omp_target_free(d, dev);
}

static void atomics() {
  const int n = 20000;
  long total = 0;
  int hist[8] = {};
  int last = -1;
  long taken = 0;
#pragma omp target teams distribute parallel for map(tofrom : total, hist[0 : 8], taken)
  for (int i = 0; i < n; ++i) {
#pragma omp atomic update
    total += i;
#pragma omp atomic update
    hist[i % 8] += 1;
    long old;
#pragma omp atomic capture
    {
      old = taken;
      taken += 1;
    }
    (void)old;
  }
  CHECK(total == static_cast<long>(n) * (n - 1) / 2, "atomic update total %ld", total);
  int bad = 0;
  for (int k = 0; k < 8; ++k) bad += hist[k] != n / 8;
  CHECK(bad == 0, "atomic histogram");
  CHECK(taken == n, "atomic capture count %ld", taken);
  (void)last;
}

static void nested_loops() {
  const int m = 48, n = 40, k = 32;
  std::vector<float> a(m * k), b(k * n), c(m * n, 0.f), want(m * n, 0.f);
  for (int i = 0; i < m * k; ++i) a[i] = static_cast<float>((i * 7) % 11) - 5.f;
  for (int i = 0; i < k * n; ++i) b[i] = static_cast<float>((i * 3) % 13) - 6.f;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float s = 0;
      for (int p = 0; p < k; ++p) s += a[i * k + p] * b[p * n + j];
      want[i * n + j] = s;
    }
  float *ap = a.data(), *bp = b.data(), *cp = c.data();
#pragma omp target teams distribute parallel for collapse(2) map(to : ap[0 : m * k], bp[0 : k * n]) \
    map(from : cp[0 : m * n])
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float s = 0;
      for (int p = 0; p < k; ++p) s += ap[i * k + p] * bp[p * n + j];
      cp[i * n + j] = s;
    }
  int wrong = 0;
  for (int i = 0; i < m * n; ++i) wrong += c[i] != want[i];
  CHECK(wrong == 0, "matrix product: %d of %d wrong", wrong, m * n);
}

static void teams_and_threads() {
  int teams = 0, threads = 0, ids_ok = 1;
  int seen[16] = {};
#pragma omp target teams num_teams(8) thread_limit(64) map(from : teams, threads) map(tofrom : seen[0 : 16])
  {
    if (omp_get_team_num() == 0) teams = omp_get_num_teams();
#pragma omp parallel num_threads(32)
    {
      if (omp_get_team_num() == 0 && omp_get_thread_num() == 0) threads = omp_get_num_threads();
      if (omp_get_thread_num() >= omp_get_num_threads()) ids_ok = 0;
    }
#pragma omp atomic update
    seen[omp_get_team_num() & 15] += 1;
  }
  CHECK(teams == 8, "num_teams(8) gave %d", teams);
  CHECK(threads == 32, "num_threads(32) gave %d", threads);
  int total = 0;
  for (int v : seen) total += v;
  CHECK(total == 8 && ids_ok, "every team ran once (%d)", total);
}


struct Pair {
  double a;
  long n;
};
#pragma omp declare reduction(pair_add : Pair : omp_out.a += omp_in.a, omp_out.n += omp_in.n) initializer(omp_priv = {0.0, 0})

struct Mapped {
  int len;
  double* data;
};
#pragma omp declare mapper(Mapped m) map(m, m.data[0 : m.len])

static void user_reduction_and_mapper() {
  const int n = 3000;
  Pair r = {0.0, 0};
#pragma omp target teams distribute parallel for reduction(pair_add : r)
  for (int i = 0; i < n; ++i) {
    r.a += 0.5 * i;
    r.n += 1;
  }
  CHECK(r.a == 0.5 * n * (n - 1) / 2 && r.n == n, "declare reduction: %.1f %ld", r.a, r.n);

  std::vector<double> v(256, 2.0);
  Mapped m = {256, v.data()};
#pragma omp target map(tofrom : m)
  {
    for (int i = 0; i < m.len; ++i) m.data[i] = m.data[i] * 3 + i;
  }
  int wrong = 0;
  for (int i = 0; i < 256; ++i) wrong += v[i] != 6.0 + i;
  CHECK(wrong == 0, "declare mapper: %d wrong", wrong);
}

static void math_functions() {
  const int n = 512;
  std::vector<double> in(n), out(n), outf(n);
  for (int i = 0; i < n; ++i) in[i] = 0.01 * (i + 1);
  double* ip = in.data();
  double* op = out.data();
  double* fp = outf.data();
#pragma omp target teams distribute parallel for map(to : ip[0 : n]) map(from : op[0 : n], fp[0 : n])
  for (int i = 0; i < n; ++i) {
    op[i] = std::sin(ip[i]) + std::cos(ip[i]) + std::exp(-ip[i]) + std::log(ip[i]) + std::sqrt(ip[i]) + std::pow(ip[i], 1.5) +
            std::tanh(ip[i]) + std::atan2(ip[i], 2.0);
    const float xf = static_cast<float>(ip[i]);
    fp[i] = static_cast<double>(std::sin(xf) + std::exp(xf) + std::sqrt(xf));
  }
  int wrong = 0;
  for (int i = 0; i < n; ++i) {
    const double x = in[i];
    const double want = std::sin(x) + std::cos(x) + std::exp(-x) + std::log(x) + std::sqrt(x) + std::pow(x, 1.5) + std::tanh(x) +
                        std::atan2(x, 2.0);
    const float xf = static_cast<float>(x);
    const double wantf = static_cast<double>(std::sin(xf) + std::exp(xf) + std::sqrt(xf));
    wrong += std::fabs(out[i] - want) > 1e-12 * (1 + std::fabs(want));
    wrong += std::fabs(outf[i] - wantf) > 1e-5 * (1 + std::fabs(wantf));
  }
  CHECK(wrong == 0, "device math library: %d of %d outside tolerance", wrong, 2 * n);
}

static void loop_construct_and_stencil() {
  const int n = 66, steps = 20;
  std::vector<double> a(n * n, 0.0), b(n * n, 0.0), ra(n * n, 0.0), rb(n * n, 0.0);
  for (int i = 0; i < n; ++i) a[i] = b[i] = ra[i] = rb[i] = 100.0;   // a hot top edge
  for (int s = 0; s < steps; ++s) {
    for (int i = 1; i < n - 1; ++i)
      for (int j = 1; j < n - 1; ++j)
        rb[i * n + j] = 0.25 * (ra[(i - 1) * n + j] + ra[(i + 1) * n + j] + ra[i * n + j - 1] + ra[i * n + j + 1]);
    ra.swap(rb);
  }
  double *ap = a.data(), *bp = b.data();
#pragma omp target data map(tofrom : ap[0 : n * n]) map(tofrom : bp[0 : n * n])
  {
    for (int s = 0; s < steps; ++s) {
#pragma omp target teams loop collapse(2)
      for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
          bp[i * n + j] = 0.25 * (ap[(i - 1) * n + j] + ap[(i + 1) * n + j] + ap[i * n + j - 1] + ap[i * n + j + 1]);
      double* t = ap;
      ap = bp;
      bp = t;
    }
  }
  // The pointers swapped steps times; the data region mapped the originals.
  int wrong = 0;
  const std::vector<double>& result = (steps % 2) ? b : a;
  for (int i = 0; i < n * n; ++i) wrong += result[i] != ra[i];
  CHECK(wrong == 0, "jacobi with target teams loop: %d of %d wrong", wrong, n * n);
}

static void always_and_present() {
  int x = 1;
#pragma omp target enter data map(to : x)
  x = 2;                                   // the host changes; the device copy stays 1
  int seen = 0;
#pragma omp target map(present, to : x) map(from : seen)
  { seen = x; }
  CHECK(seen == 1, "map(present) read %d, not the device copy", seen);
#pragma omp target map(always, to : x) map(from : seen)
  { seen = x; }
  CHECK(seen == 2, "map(always, to) read %d", seen);
#pragma omp target exit data map(delete : x)
  CHECK(omp_target_is_present(&x, omp_get_default_device()) == 0, "deleted mapping is still present");
}

static void asynchronous() {
  const int n = 1 << 14;
  std::vector<int> a(n, 1), b(n, 2), c(n, 0);
  int *ap = a.data(), *bp = b.data(), *cp = c.data();
  int token1 = 0, token2 = 0;
#pragma omp target nowait depend(out : token1) map(tofrom : ap[0 : n])
  for (int i = 0; i < n; ++i) ap[i] += 10;
#pragma omp target nowait depend(out : token2) map(tofrom : bp[0 : n])
  for (int i = 0; i < n; ++i) bp[i] *= 5;
#pragma omp target nowait depend(in : token1, token2) map(to : ap[0 : n], bp[0 : n]) map(from : cp[0 : n])
  for (int i = 0; i < n; ++i) cp[i] = ap[i] + bp[i];
#pragma omp taskwait
  int wrong = 0;
  for (int i = 0; i < n; ++i) wrong += c[i] != 21;
  CHECK(wrong == 0, "dependent nowait targets: %d wrong", wrong);
}

static void every_device() {
  const int n = omp_get_num_devices();
  int ran_on_each = 0;
  for (int d = 0; d < n; ++d) {
    int id = -1;
#pragma omp target device(d) map(from : id)
    { id = 100 + omp_get_device_num(); }
    ran_on_each += id == 100 + d;
  }
  CHECK(ran_on_each == n, "device(d) reached %d of %d devices", ran_on_each, n);
}

static void device_printf() {
  std::fflush(stdout);
#pragma omp target
  { std::printf("printf from the device: %d %.2f\n", 42, 2.5); }
}

int main() {
  devices();
  saxpy();
  reductions();
  data_regions();
  declare_target();
  device_memory();
  atomics();
  nested_loops();
  teams_and_threads();
  user_reduction_and_mapper();
  math_functions();
  loop_construct_and_stencil();
  always_and_present();
  asynchronous();
  every_device();
  device_printf();
  std::printf("%d checks, %d failed\n", g_checks, g_failed);
  return g_failed ? 1 : 0;
}
