// How many bytes a lane of an LDS instruction moves (vgpu/amd_memtraffic.hpp), checked two ways: a table of names whose
// answer is written out here from the ISA's own naming, and every ds_ name the decoder knows, so that an instruction added to
// the decoder without a decision about its bytes fails here instead of reading as zero.
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>

#include "vgpu/amd_memtraffic.hpp"
#include "vtest.hpp"

using vgpu::amd::gcn::ds_lane_traffic;
using vgpu::amd::gcn::LaneTraffic;

namespace {

struct Case {
  const char* name;
  uint32_t read, written;
};

// The access size is in the name; two addresses doubles it; an atomic is the written bytes of its type.
const Case kCases[] = {
    {"ds_read_b32", 4, 0},          {"ds_read_b64", 8, 0},          {"ds_read_b96", 12, 0},          {"ds_read_b128", 16, 0},
    {"ds_read_u8", 1, 0},           {"ds_read_i8", 1, 0},           {"ds_read_u16", 2, 0},           {"ds_read_i16", 2, 0},
    {"ds_read_u8_d16_hi", 1, 0},    {"ds_read_u16_d16", 2, 0},      {"ds_read2_b32", 8, 0},          {"ds_read2_b64", 16, 0},
    {"ds_read2st64_b32", 8, 0},     {"ds_read2st64_b64", 16, 0},    {"ds_read_addtid_b32", 4, 0},    {"ds_read_b64_tr_b16", 8, 0},
    {"ds_load_b32", 4, 0},          {"ds_load_b128", 16, 0},        {"ds_load_u8", 1, 0},            {"ds_load_2addr_b32", 8, 0},
    {"ds_load_2addr_stride64_b64", 16, 0},                          {"ds_load_b128_tr_b6", 16, 0},   {"ds_load_tr6_b96", 12, 0},
    {"ds_load_tr16_b128", 16, 0},   {"ds_load_b64_tr_b4", 8, 0},
    {"ds_write_b8", 0, 1},          {"ds_write_b16", 0, 2},         {"ds_write_b32", 0, 4},          {"ds_write_b64", 0, 8},
    {"ds_write_b96", 0, 12},        {"ds_write_b128", 0, 16},       {"ds_write_b8_d16_hi", 0, 1},    {"ds_write_b16_d16_hi", 0, 2},
    {"ds_write2_b32", 0, 8},        {"ds_write2_b64", 0, 16},       {"ds_write2st64_b64", 0, 16},    {"ds_write_addtid_b32", 0, 4},
    {"ds_store_b32", 0, 4},         {"ds_store_2addr_b64", 0, 16},  {"ds_store_b8_d16_hi", 0, 1},
    {"ds_add_u32", 0, 4},           {"ds_add_rtn_u32", 0, 4},       {"ds_add_u64", 0, 8},            {"ds_add_f32", 0, 4},
    {"ds_add_f64", 0, 8},           {"ds_max_num_f32", 0, 4},       {"ds_min_rtn_f64", 0, 8},        {"ds_inc_u32", 0, 4},
    {"ds_cmpst_b64", 0, 8},         {"ds_cmpstore_rtn_f32", 0, 4},  {"ds_mskor_b32", 0, 4},          {"ds_wrxchg_rtn_b64", 0, 8},
    {"ds_wrxchg2_rtn_b32", 0, 8},   {"ds_storexchg_rtn_b32", 0, 4}, {"ds_pk_add_f16", 0, 4},         {"ds_pk_add_rtn_bf16", 0, 4},
    {"ds_cond_sub_u32", 0, 4},      {"ds_sub_clamp_rtn_u32", 0, 4}, {"ds_wrap_rtn_b32", 0, 4},
};

// The ones that move nothing to or from LDS memory.
const std::set<std::string> kNoData = {
    "ds_nop", "ds_swizzle_b32", "ds_permute_b32", "ds_bpermute_b32", "ds_bpermute_fi_b32", "ds_append", "ds_consume", "ds_ordered_count",
    "ds_param_load", "ds_direct_load", "ds_condxchg32_rtn_b64", "ds_add_gs_reg_rtn", "ds_sub_gs_reg_rtn",
};

bool no_data_by_prefix(const std::string& n) {
  return n.rfind("ds_gws", 0) == 0 || n.rfind("ds_bvh_", 0) == 0 || n.find("barrier") != std::string::npos;
}

// Every "ds_..." string literal in the decoder's and the RDNA tables' sources.
std::set<std::string> source_names() {
  std::set<std::string> names;
  const std::filesystem::path dir = std::filesystem::path(VGPU_SOURCE_DIR) / "amd" / "src";
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    const std::string f = entry.path().filename().string();
    if (f != "gcn_decode.cpp" && f != "gcn_tier.cpp" && f.rfind("rdna_ops", 0) != 0) continue;
    std::ifstream in(entry.path(), std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), {});
    for (size_t at = text.find("\"ds_"); at != std::string::npos; at = text.find("\"ds_", at + 1)) {
      size_t end = at + 1;
      while (end < text.size() && (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_')) ++end;
      if (end >= text.size() || text[end] != '"') continue;
      std::string n = text.substr(at + 1, end - at - 1);
      for (char& c : n) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      names.insert(n);
    }
  }
  return names;
}

}  // namespace

VTEST(each_named_access_moves_the_bytes_its_name_says) {
  for (const Case& c : kCases) {
    const LaneTraffic t = ds_lane_traffic(c.name);
    if (t.read != c.read || t.written != c.written)
      throw vtest::Failure(std::string(c.name) + ": read " + std::to_string(t.read) + " written " + std::to_string(t.written) +
                           ", expected " + std::to_string(c.read) + " and " + std::to_string(c.written));
  }
}

VTEST(the_name_without_ds_prefix_gives_the_same_answer) {
  const LaneTraffic a = ds_lane_traffic("ds_read2_b64"), b = ds_lane_traffic("read2_b64");
  VCHECK_EQ(a.read, b.read);
  VCHECK_EQ(a.written, b.written);
}

VTEST(every_ds_instruction_the_decoder_knows_is_decided) {
  const std::set<std::string> names = source_names();
  VCHECK(names.size() > 150);   // the sources were found and read
  for (const std::string& n : names) {
    // (A few string literals are prefixes the decoder builds names from, not instructions.)
    if (n == "ds_store" || n == "ds_write" || n == "ds_gws_" || n == "ds_load" || n == "ds_read") continue;
    const LaneTraffic t = ds_lane_traffic(n);
    const bool moves = t.read || t.written;
    const bool expect_none = kNoData.count(n) || no_data_by_prefix(n);
    if (moves == expect_none) throw vtest::Failure(n + (moves ? " moves bytes but is listed as moving none" : " is not listed as moving none, yet counts none"));
    if (t.read && t.written) throw vtest::Failure(n + " is counted as both read and written");
  }
}

VTEST_MAIN
