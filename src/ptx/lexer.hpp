// PTX tokenizer (internal to the parser).
#pragma once

#include <string>
#include <vector>

namespace vgpu::ptx {

struct Token {
  enum class Kind { Word, Punct, End };
  Kind kind = Kind::End;
  std::string text;  // word text, or single punct char
  size_t line = 0;
};

// Splits PTX source into tokens. Words are runs of [A-Za-z0-9_$%.]; everything
// else meaningful is a single-char punct. Comments (// and /* */) are skipped.
std::vector<Token> lex(const std::string& src);

}  // namespace vgpu::ptx
