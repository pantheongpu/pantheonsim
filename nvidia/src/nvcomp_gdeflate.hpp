// GDeflate for the simulator's nvCOMP: DEFLATE with DEFLATE64's long copies,
// its codes dealt round-robin over 32 sub-streams packed into interleaved
// 32-bit words, so that a 32-lane SIMD group decodes one code per lane per
// round. Written from NVIDIA's published specification, the Internet-Draft
// "GDEFLATE bitstream specification" (draft-uralsky-gdeflate-00, July 2024).
//
// Where the draft leaves the layout open, the answer is the one that decodes
// every stream NVIDIA's nvCOMP 5.3 produced on an RTX 3060, word for word:
//   - the decoder tops up every lane holding fewer than 32 bits at the end of
//     each round (and so once more after the last one), taking words in lane
//     order; the stream is exactly as long as those reads;
//   - a copy's distance code is decoded by its lane in the round after its
//     length, and lanes past an end-of-block code still decode distances that
//     are due; distances due after the end-of-block round get a round of
//     their own.
// nvCOMP writes one dynamic-Huffman block per chunk, of any size, with
// copies reaching back up to 64 KiB, and so does this encoder. The decoder
// also reads stored and fixed-Huffman blocks and streams of several blocks,
// laid out as the draft describes.
#ifndef VGPU_NVCOMP_GDEFLATE_HPP
#define VGPU_NVCOMP_GDEFLATE_HPP

#include "nvcomp_codecs.hpp"

namespace vgpu::codec {

// NVIDIA's bound (nvcompBatchedGdeflateCompressGetMaxOutputChunkSize on an
// RTX 3060): (2n + 288) rounded down to a whole word.
size_t gdeflate_bound(size_t n);
// effort as Deflate's: 0 is Huffman coding alone, 1..5 search harder.
Bytes gdeflate_compress(const uint8_t* in, size_t n, int effort);
// For tests: the tokens split evenly over blocks of the given types (0
// stored, 1 fixed Huffman, 2 dynamic Huffman). A stored block must hold no
// more than 65535 bytes.
Bytes gdeflate_compress_blocks(const uint8_t* in, size_t n, int effort, const std::vector<int>& types);
Result gdeflate_decompress(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced);

}  // namespace vgpu::codec

#endif  // VGPU_NVCOMP_GDEFLATE_HPP
