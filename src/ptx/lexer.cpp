#include "lexer.hpp"

#include <cctype>

#include "vgpu/error.hpp"

namespace vgpu::ptx {
namespace {

bool word_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' || c == '%' || c == '.';
}

}  // namespace

std::vector<Token> lex(const std::string& src) {
  std::vector<Token> out;
  size_t line = 1;
  size_t i = 0;
  const size_t n = src.size();
  while (i < n) {
    char c = src[i];
    if (c == '\n') {
      ++line;
      ++i;
      continue;
    }
    if (std::isspace(static_cast<unsigned char>(c))) {
      ++i;
      continue;
    }
    if (c == '/' && i + 1 < n && src[i + 1] == '/') {
      while (i < n && src[i] != '\n') ++i;
      continue;
    }
    if (c == '/' && i + 1 < n && src[i + 1] == '*') {
      size_t start_line = line;
      i += 2;
      while (i + 1 < n && !(src[i] == '*' && src[i + 1] == '/')) {
        if (src[i] == '\n') ++line;
        ++i;
      }
      if (i + 1 >= n) throw Error::make(Err::PtxParse, "line ", start_line, ": unterminated /* comment");
      i += 2;
      continue;
    }
    if (c == '"') {
      // String literal (e.g. .pragma "nounroll"). Kept as one token, quotes and all.
      size_t start = i++;
      while (i < n && src[i] != '"') {
        if (src[i] == '\n') ++line;
        ++i;
      }
      if (i >= n) throw Error::make(Err::PtxParse, "line ", line, ": unterminated string literal");
      ++i;
      out.push_back({Token::Kind::Word, src.substr(start, i - start), line});
      continue;
    }
    if (word_char(c)) {
      size_t start = i;
      while (i < n) {
        if (word_char(src[i])) {
          ++i;
          continue;
        }
        // A *doubled* colon belongs to a cache-hint qualifier (.L2::128B,
        // .L1::no_allocate) and is part of the opcode; a single one ends a
        // label. Splitting on the pair would tear an opcode in half, which is
        // how ggml's flash-attention kernels failed to parse at all.
        if (src[i] == ':' && i + 2 < n && src[i + 1] == ':' && word_char(src[i + 2])) {
          i += 2;
          continue;
        }
        break;
      }
      out.push_back({Token::Kind::Word, src.substr(start, i - start), line});
      continue;
    }
    // '|' appears in shuffle/vote destinations ("d|p"); '%' only inside words.
    static const std::string puncts = ",;:()[]{}@+<>!-=*|";
    if (puncts.find(c) != std::string::npos) {
      out.push_back({Token::Kind::Punct, std::string(1, c), line});
      ++i;
      continue;
    }
    throw Error::make(Err::PtxParse, "line ", line, ": unexpected character '", std::string(1, c), "'");
  }
  out.push_back({Token::Kind::End, "", line});
  return out;
}

}  // namespace vgpu::ptx
