// sass::unsupported keeps the instruction words it has already seen decode and run (per thread), so the next cubin
// of a process -- the kernels of a PyTorch build repeat the same words by the million -- is not decoded word for
// word again. The table must change nothing a caller can see: the same answer, and for a refusal the same text with
// the same section and offset, whatever it has seen before. Each test below would fail if the table answered for a
// word it had not seen (a different architecture, a word differing in one bit of either half), remembered a
// refusal, or hid the instruction VGPU_SASS_REFUSE names.
//
// The cubins are built here: an ELF header, a section string table and one .text.k section of the words under test.
// Words are found by searching (the corpora of nvidia/tests/data/sass, and single-bit changes of their words),
// never written down, so the test follows the decoder.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/sass/exec.hpp"
#include "vgpu/sass/sass.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace vgpu::sass {
bool runs_instr(const Instr& ins);   // src/sass/exec.cpp
}

namespace {

// The table of words is under test here, not the file of images (a hit there would skip the decoding these tests watch).
[[maybe_unused]] const int kNoFile = setenv("VGPU_SASS_CACHE", "0", 1);

template <class T>
void put(std::string& s, T v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof v);
}

// A cubin for sm_<sm> whose only code is .text.k holding `words`, in order.
std::string make_cubin(int sm, const std::vector<sass::Word>& words) {
  const std::string names = std::string("\0.shstrtab\0.text.k\0", 19);   // .shstrtab at 1, .text.k at 11
  std::string out(64, '\0');
  const uint64_t names_off = out.size();
  out += names;
  while (out.size() % 16) out.push_back('\0');
  const uint64_t text_off = out.size();
  for (const sass::Word& w : words) {
    put(out, w.lo);
    put(out, w.hi);
  }
  const uint64_t shoff = out.size();
  const auto shdr = [&](uint32_t name, uint32_t type, uint64_t off, uint64_t size) {
    put<uint32_t>(out, name);
    put<uint32_t>(out, type);
    put<uint64_t>(out, 0);
    put<uint64_t>(out, 0);
    put<uint64_t>(out, off);
    put<uint64_t>(out, size);
    put<uint32_t>(out, 0);
    put<uint32_t>(out, 0);
    put<uint64_t>(out, 16);
    put<uint64_t>(out, 0);
  };
  shdr(0, 0, 0, 0);
  shdr(1, 3, names_off, names.size());
  shdr(11, 1, text_off, 16 * words.size());
  out[0] = 0x7f;
  out[1] = 'E';
  out[2] = 'L';
  out[3] = 'F';
  out[4] = 2;
  out[5] = 1;
  out[6] = 1;
  out[7] = 0x41;
  out[8] = 8;
  const uint16_t type = 2, machine = 190, shentsize = 64, shnum = 3, shstrndx = 1;
  const uint32_t flags = 0x06000002u | (static_cast<uint32_t>(sm) << 8);
  std::memcpy(&out[16], &type, 2);
  std::memcpy(&out[18], &machine, 2);
  std::memcpy(&out[0x28], &shoff, 8);
  std::memcpy(&out[0x30], &flags, 4);
  std::memcpy(&out[0x3a], &shentsize, 2);
  std::memcpy(&out[0x3c], &shnum, 2);
  std::memcpy(&out[0x3e], &shstrndx, 2);
  return out;
}

std::string unsupported(int sm, const std::vector<sass::Word>& words) {
  const std::string c = make_cubin(sm, words);
  return sass::unsupported(reinterpret_cast<const uint8_t*>(c.data()), c.size());
}

// What unsupported() decides about one word, by the decoder and executor directly.
bool runs(int sm, const sass::Word& w, size_t at = 0) {
  try {
    return sass::runs_instr(sass::decode(w, 16 * at, sm));
  } catch (const Error&) {
    return false;
  }
}

std::vector<sass::Word> corpus(const char* name) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/nvidia/tests/data/sass/" + name + ".txt");
  if (!in) throw vtest::Failure(std::string("no corpus ") + name);
  std::vector<sass::Word> out;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    sass::Word w;
    w.hi = std::stoull(line.substr(0, 16), nullptr, 16);
    w.lo = std::stoull(line.substr(16, 16), nullptr, 16);
    out.push_back(w);
  }
  return out;
}

// Words that run on sm_86, a handful, spread through the corpus.
std::vector<sass::Word> good_words(size_t n, int sm = 86) {
  std::vector<sass::Word> all = corpus(("sm_" + std::to_string(sm)).c_str()), out;
  for (size_t i = 0; i < all.size() && out.size() < n; i += 97)
    if (runs(sm, all[i])) out.push_back(all[i]);
  return out;
}

sass::Word unknown_opcode_word() {
  sass::Word w;
  w.lo = 0xfff;   // an opcode no decoder has
  return w;
}

std::string expected_refusal(int sm, size_t at, const sass::Word& w) {
  char b[64];
  std::snprintf(b, sizeof b, "sm_%d k+0x%zx: ", sm, 16 * at);
  try {
    const sass::Instr ins = sass::decode(w, 16 * at, sm);
    return std::string(b) + sass::to_text(ins);
  } catch (const Error& e) {
    return std::string(b) + e.message();
  }
}

// The table is per thread; a fresh thread starts with an empty one, so each test sees only what it put there.
template <class F>
void on_fresh_thread(F&& f) {
  std::string failure;
  std::thread t([&] {
    try {
      f();
    } catch (const std::exception& e) {
      failure = e.what();
    }
  });
  t.join();
  if (!failure.empty()) throw vtest::Failure(failure);
}

}  // namespace

VTEST(a_cubin_of_words_that_run_is_supported_however_often_it_is_asked) {
  on_fresh_thread([&] {
    unsetenv("VGPU_SASS_REFUSE");
    const std::vector<sass::Word> good = good_words(40);
    VCHECK(good.size() >= 20);
    for (int round = 0; round < 3; ++round) VCHECK_EQ(unsupported(86, good), std::string());
  });
}

VTEST(a_refusal_names_the_same_place_whatever_was_seen_before) {
  on_fresh_thread([&] {
    unsetenv("VGPU_SASS_REFUSE");
    std::vector<sass::Word> good = good_words(30);
    VCHECK(unsupported(86, good).empty());   // all of these are now known
    std::vector<sass::Word> words = good;
    words.insert(words.begin() + 7, unknown_opcode_word());
    const std::string want = expected_refusal(86, 7, unknown_opcode_word());
    VCHECK(want.find("unknown opcode") != std::string::npos);
    VCHECK_EQ(unsupported(86, words), want);
    VCHECK_EQ(unsupported(86, words), want);   // a refusal is not remembered as a pass
    words.insert(words.begin() + 3, unknown_opcode_word());   // an earlier one is the one named
    VCHECK_EQ(unsupported(86, words), expected_refusal(86, 3, unknown_opcode_word()));
  });
}

VTEST(an_instruction_the_executor_refuses_is_named_by_its_text) {
  on_fresh_thread([&] {
    unsetenv("VGPU_SASS_REFUSE");
    // A word the decoder reads but the executor does not run (a texture form, an LDGMC): found in the corpora.
    sass::Word refused;
    int refused_sm = 0;
    for (const char* name : {"sm_86", "sm_90", "sm_100", "sm_120", "sm_80", "sm_75", "sm_89"}) {
      const int sm = std::atoi(name + 3);
      for (const sass::Word& w : corpus(name)) {
        if (refused_sm) break;
        try {
          if (!sass::runs_instr(sass::decode(w, 0, sm))) {
            refused = w;
            refused_sm = sm;
          }
        } catch (const Error&) {
        }
      }
    }
    const bool found = refused_sm != 0;
    VCHECK(found);
    std::vector<sass::Word> words = good_words(10, refused_sm);
    VCHECK(words.size() == 10);
    VCHECK(unsupported(refused_sm, words).empty());
    words.push_back(refused);
    VCHECK_EQ(unsupported(refused_sm, words), expected_refusal(refused_sm, 10, refused));
    VCHECK_EQ(unsupported(refused_sm, words), expected_refusal(refused_sm, 10, refused));
  });
}

VTEST(what_ran_on_one_architecture_is_decided_again_for_another) {
  on_fresh_thread([&] {
    unsetenv("VGPU_SASS_REFUSE");
    // A word that runs on one architecture and is refused on another.
    const std::vector<sass::Word> all = corpus("sm_86");
    const std::vector<sass::Word> more = corpus("sm_120");
    bool found = false;
    sass::Word w;
    int ok_sm = 0, bad_sm = 0;
    std::vector<sass::Word> pool = all;
    pool.insert(pool.end(), more.begin(), more.end());
    for (const sass::Word& c : pool) {
      for (int a : {75, 80, 86, 89, 90, 100, 120})
        for (int b : {75, 80, 86, 89, 90, 100, 120})
          if (!found && a != b && runs(a, c) && !runs(b, c)) {
            found = true;
            w = c;
            ok_sm = a;
            bad_sm = b;
          }
      if (found) break;
    }
    VCHECK(found);
    VCHECK(unsupported(ok_sm, {w}).empty());   // known on ok_sm ...
    VCHECK_EQ(unsupported(bad_sm, {w}), expected_refusal(bad_sm, 0, w));   // ... not on bad_sm
    VCHECK(unsupported(ok_sm, {w}).empty());
  });
}

VTEST(a_word_differing_in_one_bit_is_decided_by_itself) {
  on_fresh_thread([&] {
    unsetenv("VGPU_SASS_REFUSE");
    // Pairs of words one bit apart, in the low half and in the high half, that run and are refused.
    const std::vector<sass::Word> base = corpus("sm_86");
    bool lo_found = false, hi_found = false;
    sass::Word lo_ok, lo_bad, hi_ok, hi_bad;
    for (size_t i = 0; i < base.size() && !(lo_found && hi_found); ++i) {
      if (!runs(86, base[i])) continue;
      for (unsigned bit = 0; bit < 64 && !(lo_found && hi_found); ++bit) {
        sass::Word l = base[i], h = base[i];
        l.lo ^= uint64_t{1} << bit;
        h.hi ^= uint64_t{1} << bit;
        if (!lo_found && !runs(86, l)) {
          lo_found = true;
          lo_ok = base[i];
          lo_bad = l;
        }
        if (!hi_found && !runs(86, h)) {
          hi_found = true;
          hi_ok = base[i];
          hi_bad = h;
        }
      }
    }
    VCHECK(lo_found && hi_found);
    VCHECK(unsupported(86, {lo_ok, hi_ok}).empty());
    VCHECK_EQ(unsupported(86, {lo_ok, lo_bad}), expected_refusal(86, 1, lo_bad));
    VCHECK_EQ(unsupported(86, {hi_ok, hi_bad}), expected_refusal(86, 1, hi_bad));
  });
}

VTEST(vgpu_sass_refuse_still_refuses_a_word_already_seen) {
  on_fresh_thread([&] {
    unsetenv("VGPU_SASS_REFUSE");
    const std::vector<sass::Word> good = good_words(30);
    VCHECK(unsupported(86, good).empty());
    const sass::Instr first = sass::decode(good[5], 0, 86);
    setenv("VGPU_SASS_REFUSE", first.mnemonic.c_str(), 1);
    const std::string got = unsupported(86, good);
    unsetenv("VGPU_SASS_REFUSE");
    VCHECK(!got.empty());
    VCHECK(got.rfind("sm_86 k+0x", 0) == 0);
    VCHECK(unsupported(86, good).empty());   // and the variable's absence passes them again
  });
}

VTEST_MAIN
