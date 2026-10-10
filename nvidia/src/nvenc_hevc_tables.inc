// Part of nvenc_hevc_enc.cpp (included inside its anonymous namespace): the tables of ITU-T H.265 the encoder needs.

// ---- CABAC contexts: numbering, and the initValues of Tables 9-5 to 9-37 for the three initTypes ----------------------------------------

enum HevcCtx {
  kCtxSplitCu = 0,         // 3
  kCtxSkip = 3,            // 3
  kCtxPredMode = 6,        // 1
  kCtxPartMode = 7,        // 4
  kCtxPrevIntraLuma = 11,  // 1
  kCtxChromaPred = 12,     // 1
  kCtxMergeFlag = 13,      // 1
  kCtxMergeIdx = 14,       // 1
  kCtxInterPredIdc = 15,   // 5
  kCtxRefIdx = 20,         // 2
  kCtxMvp = 22,            // 1
  kCtxRqtRoot = 23,        // 1
  kCtxSplitTransform = 24, // 3
  kCtxCbfLuma = 27,        // 2
  kCtxCbfChroma = 29,      // 5
  kCtxMvdG0 = 34,          // 1
  kCtxMvdG1 = 35,          // 1
  kCtxCuQpDelta = 36,      // 2
  kCtxTransformSkip = 38,  // 2: luma, chroma
  kCtxLastX = 40,          // 18
  kCtxLastY = 58,          // 18
  kCtxCsbf = 76,           // 4
  kCtxSig = 80,            // 44
  kCtxGreater1 = 124,      // 24
  kCtxGreater2 = 148,      // 6
  kCtxSaoMerge = 154,      // 1
  kCtxSaoType = 155,       // 1
  kCtxCount = 156
};

struct HevcInit {
  uint8_t v[3][kCtxCount];
  HevcInit() {
    std::memset(v, 154, sizeof v);
    auto put = [&](int base, int type, std::initializer_list<int> vals) {
      int i = 0;
      for (int x : vals) v[type][base + i++] = static_cast<uint8_t>(x);
    };
    for (int t = 0; t < 3; ++t) {
      put(kCtxSaoMerge, t, {153});
      put(kCtxTransformSkip, t, {139, 139});
      put(kCtxCuQpDelta, t, {154, 154});
    }
    put(kCtxSaoType, 0, {200});
    put(kCtxSaoType, 1, {185});
    put(kCtxSaoType, 2, {160});
    put(kCtxSplitCu, 0, {139, 141, 157});
    put(kCtxSplitCu, 1, {107, 139, 126});
    put(kCtxSplitCu, 2, {107, 139, 126});
    put(kCtxSkip, 1, {197, 185, 201});
    put(kCtxSkip, 2, {197, 185, 201});
    put(kCtxPredMode, 1, {149});
    put(kCtxPredMode, 2, {134});
    put(kCtxPartMode, 0, {184, 154, 139, 154});
    put(kCtxPartMode, 1, {154, 139, 154, 154});
    put(kCtxPartMode, 2, {154, 139, 154, 154});
    put(kCtxPrevIntraLuma, 0, {184});
    put(kCtxPrevIntraLuma, 1, {154});
    put(kCtxPrevIntraLuma, 2, {183});
    put(kCtxChromaPred, 0, {63});
    put(kCtxChromaPred, 1, {152});
    put(kCtxChromaPred, 2, {152});
    put(kCtxMergeFlag, 1, {110});
    put(kCtxMergeFlag, 2, {154});
    put(kCtxMergeIdx, 1, {122});
    put(kCtxMergeIdx, 2, {137});
    put(kCtxInterPredIdc, 1, {95, 79, 63, 31, 31});
    put(kCtxInterPredIdc, 2, {95, 79, 63, 31, 31});
    put(kCtxRefIdx, 1, {153, 153});
    put(kCtxRefIdx, 2, {153, 153});
    put(kCtxMvp, 1, {168});
    put(kCtxMvp, 2, {168});
    put(kCtxRqtRoot, 1, {79});
    put(kCtxRqtRoot, 2, {79});
    put(kCtxSplitTransform, 0, {153, 138, 138});
    put(kCtxSplitTransform, 1, {124, 138, 94});
    put(kCtxSplitTransform, 2, {224, 167, 122});
    put(kCtxCbfLuma, 0, {111, 141});
    put(kCtxCbfLuma, 1, {153, 111});
    put(kCtxCbfLuma, 2, {153, 111});
    put(kCtxCbfChroma, 0, {94, 138, 182, 154, 154});
    put(kCtxCbfChroma, 1, {149, 107, 167, 154, 154});
    put(kCtxCbfChroma, 2, {149, 92, 167, 154, 154});
    put(kCtxMvdG0, 1, {140});
    put(kCtxMvdG0, 2, {169});
    put(kCtxMvdG1, 1, {198});
    put(kCtxMvdG1, 2, {198});
    const int last0[18] = {110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63};
    const int last1[18] = {125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108};
    const int last2[18] = {125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93};
    for (int i = 0; i < 18; ++i) {
      v[0][kCtxLastX + i] = v[0][kCtxLastY + i] = static_cast<uint8_t>(last0[i]);
      v[1][kCtxLastX + i] = v[1][kCtxLastY + i] = static_cast<uint8_t>(last1[i]);
      v[2][kCtxLastX + i] = v[2][kCtxLastY + i] = static_cast<uint8_t>(last2[i]);
    }
    put(kCtxCsbf, 0, {91, 171, 134, 141});
    put(kCtxCsbf, 1, {121, 140, 61, 154});
    put(kCtxCsbf, 2, {121, 140, 61, 154});
    put(kCtxSig, 0, {111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 107,
                     125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153, 136, 139, 111, 136, 139, 111});
    put(kCtxSig, 1, {155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 166,
                     183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167, 151, 183, 140, 151, 183, 140});
    put(kCtxSig, 2, {170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 166,
                     183, 140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121, 167, 151, 183, 140, 151, 183, 140});
    put(kCtxGreater1, 0, {140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166, 182, 140, 227, 122, 197});
    put(kCtxGreater1, 1, {154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194, 166, 167, 154, 167, 137, 182});
    put(kCtxGreater1, 2, {154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 122, 169, 208, 166, 167, 154, 152, 167, 182});
    put(kCtxGreater2, 0, {138, 153, 136, 167, 152, 152});
    put(kCtxGreater2, 1, {107, 167, 91, 122, 107, 167});
    put(kCtxGreater2, 2, {107, 167, 91, 107, 107, 167});
  }
};
const HevcInit& hevc_init() {
  static const HevcInit t;
  return t;
}

// ---- transforms (8.6.4.2): the 32x32 DCT matrix, whose rows 0, 32/N, 2 * 32/N ... and first N columns are the N-point matrix ----------

// The 32 values the matrix is made of: |T[k][n]| = kCoef[(k * (2n + 1)) mod 128 folded into 0..32] up to sign (cosine symmetries).
constexpr int kDctCoef[33] = {64, 90, 90, 90, 89, 88, 87, 85, 83, 82, 80, 78, 75, 73, 70, 67, 64, 61, 57, 54, 50, 46, 43, 38, 36, 31, 25, 22, 18, 13, 9, 4, 0};

struct DctMatrix {
  int8_t t[32][32];
  DctMatrix() {
    for (int k = 0; k < 32; ++k)
      for (int n = 0; n < 32; ++n) {
        if (k == 0) {
          t[k][n] = 64;
          continue;
        }
        int a = (k * (2 * n + 1)) % 128;   // the angle in units of pi/64, period 128
        if (a > 64) a = 128 - a;           // cos(2 pi - x) = cos x
        int sign = 1;
        if (a > 32) {                      // cos(pi - x) = -cos x
          a = 64 - a;
          sign = -1;
        }
        t[k][n] = static_cast<int8_t>(sign * kDctCoef[a]);
      }
  }
};
const DctMatrix& dct_matrix() {
  static const DctMatrix m;
  return m;
}

constexpr int kDst4[4][4] = {{29, 55, 74, 84}, {74, 74, 0, -74}, {84, -29, -74, 55}, {55, -84, 74, -29}};

// ---- scans (6.5.3 to 6.5.5): ScanOrder[log2BlockSize][scanIdx][position] = (x, y); scanIdx 0 up-right diagonal, 1 horizontal, 2 vertical -----------

struct ScanTables {
  uint8_t x[4][3][64], y[4][3][64];   // block sizes 1 (log2 0), 2, 4, 8 -- 1x1 .. 8x8 (a 32x32 transform has 8x8 sub-blocks)
  ScanTables() {
    for (int l = 0; l < 4; ++l) {
      const int blk = 1 << l;
      int i = 0, xx = 0, yy = 0;
      bool stop = false;
      while (!stop) {
        while (yy >= 0) {
          if (xx < blk && yy < blk) {
            x[l][0][i] = static_cast<uint8_t>(xx);
            y[l][0][i] = static_cast<uint8_t>(yy);
            ++i;
          }
          --yy;
          ++xx;
        }
        yy = xx;
        xx = 0;
        if (i >= blk * blk) stop = true;
      }
      i = 0;
      for (int b = 0; b < blk; ++b)
        for (int a = 0; a < blk; ++a) {   // horizontal: row by row; vertical: column by column
          x[l][1][i] = static_cast<uint8_t>(a);
          y[l][1][i] = static_cast<uint8_t>(b);
          x[l][2][i] = static_cast<uint8_t>(b);
          y[l][2][i] = static_cast<uint8_t>(a);
          ++i;
        }
    }
  }
};
const ScanTables& scan_tables() {
  static const ScanTables t;
  return t;
}

// ---- intra prediction (8.4.4.2.6) -------------------------------------------------------------------------------------------------------------

constexpr int kIntraAngle[35] = {0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32};
// invAngle for modes 11..25
constexpr int kInvAngle[35] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// Table 8-10 (ChromaArrayType 1): QpC from qPi
inline int chroma_qp(int qpi) {
  static const int8_t t[14] = {29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37};
  if (qpi < 30) return qpi;
  if (qpi >= 44) return qpi - 6;
  return t[qpi - 30];
}

constexpr int kLevelScale[6] = {40, 45, 51, 57, 64, 72};
constexpr int kQuantScale[6] = {26214, 23302, 20560, 18396, 16384, 14564};

// ---- the deblocking filter (8.7.2): beta' and tc' by Q (Tables 8-12) ---------------------------------------------------------------------------

constexpr uint8_t kBetaTable[52] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                    16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64};
constexpr uint8_t kTcTable[54] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24};
