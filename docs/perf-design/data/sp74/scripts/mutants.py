#!/usr/bin/env python3
"""Round sp74: mutants of the tier's row handlers for the 64-bit forms and SDWA (the end of amd/src/gcn_tier.cpp). Each is one edit in a
build tree (put back afterwards); test_amd_gcn_tier_same_bits is rebuilt and run on the instructions the mutant names (VGPU_TIER_ONLY),
and must fail.

  mutants.py --src <tree> --build <tree>/build [--only N,M] [--lock /tmp/pantheonsim-builder.lock]

The tree is best one configured with -DVGPU_LTO=OFF (a mutant is then one compile and a link).
"""
import argparse
import os
import subprocess

F = "amd/src/gcn_tier.cpp"
# (what it does, the text, the text instead, the instructions to run)
M = [
    ("min/max: two zeros of opposite sign are done by the row", "    return mask4(_mm256_cmp_pd(x, y, _CMP_UNORD_Q)) | mask4(opposite);", "    return mask4(_mm256_cmp_pd(x, y, _CMP_UNORD_Q));", "_f64 v"),
    ("min/max: a NaN operand is done by the row", "    return mask4(_mm256_cmp_pd(x, y, _CMP_UNORD_Q)) | mask4(opposite);", "    return mask4(opposite);", "_f64 v"),
    ("min/max: the maximum is the minimum", "Max ? _mm256_max_pd(x, y) : _mm256_min_pd(x, y)", "Max ? _mm256_min_pd(x, y) : _mm256_max_pd(x, y)", "_f64 v"),
    ("min/max: the lanes not done are ignored", "  if (bad & ex) return h_f64_avx2(o, R);\n  put_pair(o, R, rl, rh, ex, n);\n  return true;\n}\nint h_dmin_rows", "  put_pair(o, R, rl, rh, ex, n);\n  return true;\n}\nint h_dmin_rows", "_f64 v"),
    ("a pair's negation is dropped", "      for (unsigned i = 0; i < n; ++i) th[i] = (w.hi[i] & keep) ^ flip;", "      for (unsigned i = 0; i < n; ++i) th[i] = (w.hi[i] & keep);", "_f64"),
    ("a pair's absolute value is dropped", "      for (unsigned i = 0; i < n; ++i) th[i] = (w.hi[i] & keep) ^ flip;", "      for (unsigned i = 0; i < n; ++i) th[i] = w.hi[i] ^ flip;", "_f64"),
    ("a constant's modifiers are dropped", "  if (s.kind == kConst) w.u = (s.value64 & keep) ^ flip;", "  if (s.kind == kConst) w.u = s.value64;", "_f64"),
    ("a scalar pair is read as one register", "  else if (s.kind == kSgpr) w.u = ((s.wide ? static_cast<uint64_t>(R.sgpr[s.index + 1]) << 32 | R.sgpr[s.index] : R.sgpr[s.index]) & keep) ^ flip;\n  else w.u = 0;   // (load64's default)", "  else if (s.kind == kSgpr) w.u = (uint64_t{R.sgpr[s.index]} & keep) ^ flip;\n  else w.u = 0;   // (load64's default)", "64"),
    ("the high row of a result is written to lanes that are off", "    merge(dlo, rl, ex, n);\n    merge(dhi, rh, ex, n);\n  } else {\n    for (uint64_t m = ex; m; m &= m - 1) {\n      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));\n      dlo[l] = rl[l], dhi[l] = rh[l];\n    }\n  }\n}\n[[gnu::always_inline]] inline __m256i set64",
     "    merge(dlo, rl, ex, n);\n    copy_row(dhi, rh, n);\n  } else {\n    for (uint64_t m = ex; m; m &= m - 1) {\n      const unsigned l = static_cast<unsigned>(__builtin_ctzll(m));\n      dlo[l] = rl[l], dhi[l] = rh[l];\n    }\n  }\n}\n[[gnu::always_inline]] inline __m256i set64", "64"),
    ("a few lanes on: the high word is not written", "      dlo[l] = rl[l], dhi[l] = rh[l];\n    }\n  }\n}\n[[gnu::always_inline]] inline __m256i set64", "      dlo[l] = rl[l];\n    }\n  }\n}\n[[gnu::always_inline]] inline __m256i set64", "64"),
    ("a 32-bit row is read as lanes 0 to 3 twice", "    x1 = _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w.lo + i + 4)));", "    x1 = x0;", "64"),
    ("one-input: a NaN result is written by the row", "    return mask4(_mm256_cmp_pd(y, y, _CMP_UNORD_Q));                               \\", "    return uint64_t{0};                                                            \\", "_f64_e"),
    ("one-input: floor is ceil", "case DFLOOR: TIER_D1(_mm256_round_pd(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)); break;", "case DFLOOR: TIER_D1(_mm256_round_pd(x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)); break;", "v_floor_f64"),
    ("one-input: ceil is floor", "case DCEIL: TIER_D1(_mm256_round_pd(x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)); break;", "case DCEIL: TIER_D1(_mm256_round_pd(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)); break;", "v_ceil_f64"),
    ("one-input: trunc is floor", "case DTRUNC: TIER_D1(_mm256_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)); break;", "case DTRUNC: TIER_D1(_mm256_round_pd(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)); break;", "v_trunc_f64"),
    ("one-input: rndne rounds toward zero", "case DRNDNE: TIER_D1(_mm256_round_pd(x, _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC)); break;", "case DRNDNE: TIER_D1(_mm256_round_pd(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)); break;", "v_rndne_f64"),
    ("one-input: the reciprocal is the value's root", "case DRCP: TIER_D1(_mm256_div_pd(one, x)); break;", "case DRCP: TIER_D1(_mm256_sqrt_pd(x)); break;", "v_rcp_f64"),
    ("one-input: the root is the reciprocal", "case DSQRT: TIER_D1(_mm256_sqrt_pd(x)); break;", "case DSQRT: TIER_D1(_mm256_div_pd(one, x)); break;", "v_sqrt_f64"),
    ("compare: the lanes that are off are in the result", "    result = by_test[test] & ex;", "    result = by_test[test];", "v_cmp"),
    ("compare: ordered ignores NaNs", "gt | eq, kind == 0 ? ~un : all,", "gt | eq, all,", "_o_f64"),
    ("compare: not-greater-or-equal is less-than", "                                  un, lt | un, eq | un,", "                                  un, lt, eq | un,", "_nge_f64"),
    ("compare: not-equal of doubles ignores NaNs", "gt | un, lt | gt | un, eq | gt | un, all};", "gt | un, lt | gt, eq | gt | un, all};", "_neq_f64"),
    ("compare: less-or-equal is less-than", "const uint64_t by_test[16] = {0, lt, eq, lt | eq, gt,", "const uint64_t by_test[16] = {0, lt, eq, lt, gt,", "_le_"),
    ("compare: unsigned is signed", "      const __m256i bias = set64(kind == 2 ? kSign : 0);", "      const __m256i bias = set64(0);", "_u64"),
    ("compare: signed is unsigned", "      const __m256i bias = set64(kind == 2 ? kSign : 0);", "      const __m256i bias = set64(kSign);", "_i64"),
    ("compare: the integers' less-than is greater-than", "        lt |= (mask4(_mm256_cmpgt_epi64(b0, a0)) | mask4(_mm256_cmpgt_epi64(b1, a1)) << 4) << i;", "        lt |= (mask4(_mm256_cmpgt_epi64(a0, b0)) | mask4(_mm256_cmpgt_epi64(a1, b1)) << 4) << i;", "_lt_"),
    ("compare: lanes 4 to 7 of each eight are lanes 0 to 3", "        eq |= (mask4(_mm256_cmp_pd(x0, y0, _CMP_EQ_OQ)) | mask4(_mm256_cmp_pd(x1, y1, _CMP_EQ_OQ)) << 4) << i;", "        eq |= (mask4(_mm256_cmp_pd(x0, y0, _CMP_EQ_OQ)) | mask4(_mm256_cmp_pd(x0, y0, _CMP_EQ_OQ)) << 4) << i;", "_eq_f64"),
    ("compare: the kind is read from the wrong bits", "kind = static_cast<unsigned>(o.off0) >> 4 & 3;", "kind = static_cast<unsigned>(o.off0) >> 5 & 3;", "v_cmp"),
    ("shift: the count is not cut to 6 bits", "*r = _mm256_sllv_epi64(b, _mm256_and_si256(a, c63)); return uint64_t{0}; });", "*r = _mm256_sllv_epi64(b, a); return uint64_t{0}; });", "v_lshlrev_b64"),
    ("shift: right is left", "*r = _mm256_srlv_epi64(b, _mm256_and_si256(a, c63)); return uint64_t{0}; });", "*r = _mm256_sllv_epi64(b, _mm256_and_si256(a, c63)); return uint64_t{0}; });", "v_lshrrev_b64"),
    ("v_lshl_add_u64: a count over 4 shifts", "_mm256_sllv_epi64(a, _mm256_andnot_si256(_mm256_cmpgt_epi64(count, c4), count))", "_mm256_sllv_epi64(a, count)", "v_lshl_add_u64"),
    ("v_lshl_add_u64: nothing is added", "        *r = _mm256_add_epi64(_mm256_sllv_epi64(a, _mm256_andnot_si256(_mm256_cmpgt_epi64(count, c4), count)), c);", "        *r = _mm256_sllv_epi64(a, _mm256_andnot_si256(_mm256_cmpgt_epi64(count, c4), count));", "v_lshl_add_u64"),
    ("v_cvt_f64_u32 converts as signed", "_mm256_add_pd(_mm256_cvtepi32_pd(_mm_xor_si128(low_words(a), top)), two31)", "_mm256_cvtepi32_pd(low_words(a))", "v_cvt_f64_u32"),
    ("v_frexp_mant_f64: a denormal is done by the row", "        return mask4(_mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_slli_epi64(a, 1), zero), e0));   // (a denormal: libm's)\n      });\n      break;\n    }\n    case L_LDEXP", "        return uint64_t{0};\n      });\n      break;\n    }\n    case L_LDEXP", "v_frexp_mant_f64"),
    ("v_frexp_mant_f64: an infinity gets a fraction's exponent", "same = _mm256_or_si256(e0, _mm256_cmpeq_epi64(e, exp));", "same = e0;", "v_frexp_mant_f64"),
    ("v_frexp_mant_f64: the exponent is 1's", "const __m256i half = set64(0x3FE0000000000000ull);", "const __m256i half = set64(0x3FF0000000000000ull);", "v_frexp_mant_f64"),
    ("v_ldexp_f64: a result past the largest number is done by the row", "_mm256_andnot_si256(_mm256_cmpgt_epi64(sum, c2046), _mm256_cmpgt_epi64(sum, zero))", "_mm256_cmpgt_epi64(sum, zero)", "v_ldexp_f64"),
    ("v_ldexp_f64: a result below the smallest normal number is done by the row", "_mm256_andnot_si256(_mm256_cmpgt_epi64(sum, c2046), _mm256_cmpgt_epi64(sum, zero))", "_mm256_andnot_si256(_mm256_cmpgt_epi64(sum, c2046), _mm256_cmpgt_epi64(c7ff, zero))", "v_ldexp_f64"),
    ("v_ldexp_f64: zero gets an exponent", "        *r = _mm256_blendv_epi8(_mm256_add_epi64(a, _mm256_slli_epi64(by, 52)), a, is_zero);", "        *r = _mm256_add_epi64(a, _mm256_slli_epi64(by, 52));", "v_ldexp_f64"),
    ("v_ldexp_f64: an infinity or a NaN is done by the row", "_mm256_and_si256(_mm256_cmpgt_epi64(e, zero), _mm256_cmpgt_epi64(c7ff, e))", "_mm256_cmpgt_epi64(e, zero)", "v_ldexp_f64"),
    ("v_ldexp_f64: the exponent is read without its sign", "        const __m256i by = _mm256_cvtepi32_epi64(low_words(b));", "        const __m256i by = _mm256_cvtepu32_epi64(low_words(b));", "v_ldexp_f64"),
    ("v_frexp_exp_i32_f64: one too few", "      const __m256i c1022 = set64(1022);", "      const __m256i c1022 = set64(1023);", "v_frexp_exp"),
    ("v_frexp_exp_i32_f64: a denormal is done by the row", "        bad |= mask4(_mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_slli_epi64(a, 1), zero), e0)) << (i + 4 * half);", "", "v_frexp_exp"),
    ("v_frexp_exp_i32_f64: an infinity has an exponent", "none = _mm256_or_si256(e0, _mm256_cmpeq_epi64(e, c7ff));", "none = e0;", "v_frexp_exp"),
    ("v_cvt_i32_f64: a NaN is the host's 0x80000000", "        out = _mm_andnot_si128(low_words(_mm256_castpd_si256(_mm256_cmp_pd(x, x, _CMP_UNORD_Q))), out);\n", "", "v_cvt_i32_f64"),
    ("v_cvt_i32_f64: a value past the largest is the host's 0x80000000", "        out = _mm_blendv_epi8(out, most, low_words(_mm256_castpd_si256(_mm256_cmp_pd(x, top, _CMP_GE_OQ))));\n", "", "v_cvt_i32_f64"),
    ("v_cvt_u32_f64: 2^31 and above is the host's signed conversion", "        out = _mm256_cvttpd_epi32(_mm256_sub_pd(x, _mm256_and_pd(high, two31)));", "        out = _mm256_cvttpd_epi32(x);", "v_cvt_u32_f64"),
    ("v_cvt_u32_f64: a negative value is not zero", "        out = _mm_and_si128(out, low_words(_mm256_castpd_si256(_mm256_cmp_pd(x, nothing, _CMP_GT_OQ))));\n", "", "v_cvt_u32_f64"),
    ("v_cvt_u32_f64: a value past the largest is not the largest", "        out = _mm_blendv_epi8(out, ones, low_words(_mm256_castpd_si256(_mm256_cmp_pd(x, top, _CMP_GE_OQ))));\n", "", "v_cvt_u32_f64"),
    ("a 32-bit result is written to lanes that are off", "  if (bad & ex) return h_l64n_avx2(o, R);\n  uint32_t* d = R.vgpr[o.dst.index];\n  if (full(ex, n)) copy_row(d, r, n);\n  else merge(d, r, ex, n);", "  if (bad & ex) return h_l64n_avx2(o, R);\n  uint32_t* d = R.vgpr[o.dst.index];\n  copy_row(d, r, n);", "_f64_e"),
    ("SDWA: a source's part is taken one byte up", "      const unsigned up = 32 - width - at, down = 32 - width;", "      const unsigned up = 32 - width - at - (at ? 8 : 0), down = 32 - width;", "_sdwa"),
    ("SDWA: a signed part is taken without its sign", "        for (unsigned i = 0; i < n; ++i) t[k][i] = static_cast<uint32_t>(static_cast<int32_t>(p[i] << up) >> down);", "        for (unsigned i = 0; i < n; ++i) t[k][i] = (p[i] << up) >> down;", "_sdwa"),
    ("SDWA: the whole register is written to lanes that are off", "  if (sel == 6) {   // (the whole register: sdwa_dst leaves the result as it is)\n    if (full(ex, n)) copy_row(d, r, n);\n    else merge(d, r, ex, n);", "  if (sel == 6) {   // (the whole register: sdwa_dst leaves the result as it is)\n    copy_row(d, r, n);", "_sdwa"),
    ("SDWA: the sign's fill is left out", "fill = o.dunused == 1 ? ~(mask | (top - 1)) : 0u,", "fill = 0u,", "_sdwa"),
    ("SDWA: the sign's fill whatever the sign", "    out[i] = (d[i] & keep) | v | ((v & top) ? fill : 0u);", "    out[i] = (d[i] & keep) | v | fill;", "_sdwa"),
    ("SDWA: the bits to keep are not kept", "keep = o.dunused == 2 ? ~mask : 0u;", "keep = 0u;", "_sdwa"),
    ("SDWA: the result is not moved to its part", "    const uint32_t v = (r[i] << shift) & mask;", "    const uint32_t v = r[i] & mask;", "_sdwa"),
    ("SDWA: a part is written to lanes that are off", "  if (full(ex, n)) copy_row(d, out, n);\n  else merge(d, out, ex, n);", "  copy_row(d, out, n);", "_sdwa"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x}
    killed = total = 0
    path = os.path.join(a.src, F)
    for i, (name, old, new, which) in enumerate(M):
        if only and i not in only:
            continue
        total += 1
        original = open(path).read()
        if original.count(old) != 1:
            print("%2d NOT APPLIED (%d matches): %s" % (i, original.count(old), name), flush=True)
            continue
        try:
            open(path, "w").write(original.replace(old, new))
            b = subprocess.run(["flock", a.lock, "nice", "-n", "19", "ninja", "-C", a.build, "-j3", "test_amd_gcn_tier_same_bits"], capture_output=True, text=True)
            if b.returncode:
                print("%2d DOES NOT BUILD: %s\n%s" % (i, name, b.stdout[-600:]), flush=True)
                continue
            t = subprocess.run([os.path.join(a.build, "test_amd_gcn_tier_same_bits")], capture_output=True, text=True, env=dict(os.environ, VGPU_TIER_ONLY=which))
            out = t.stdout + t.stderr
            why = [l.strip() for l in out.splitlines() if "runs differ" in l][:1]
            if t.returncode:
                killed += 1
                print("%2d killed: %s  [%s; %s]" % (i, name, which, why[0][:60] if why else "exit %d" % t.returncode), flush=True)
            else:
                print("%2d SURVIVED: %s  [%s]" % (i, name, which), flush=True)
        finally:
            open(path, "w").write(original)
    print("%d of %d killed" % (killed, total))


main()
