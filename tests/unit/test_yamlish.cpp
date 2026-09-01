// Unit tests for the restricted-YAML profile parser.
#include "vgpu/yamlish.hpp"

#include "vgpu/error.hpp"
#include "vtest.hpp"

using vgpu::Err;
using vgpu::Error;
using vgpu::yamlish::Value;
using vgpu::yamlish::parse;

VTEST(scalars_and_types) {
  auto doc = parse("name: hopper\ncount: 42\nneg: -7\nflag: true\noff: false\nquoted: \"a: b # c\"\n", "t");
  VCHECK_EQ(doc.map.at("name").str, "hopper");
  VCHECK_EQ(doc.map.at("count").i, 42);
  VCHECK_EQ(doc.map.at("neg").i, -7);
  VCHECK(doc.map.at("flag").b);
  VCHECK(!doc.map.at("off").b);
  VCHECK_EQ(doc.map.at("quoted").str, "a: b # c");
}

VTEST(comments_and_blank_lines) {
  auto doc = parse("# header\n\na: 1   # trailing\n\n# another\nb: two\n", "t");
  VCHECK_EQ(doc.map.at("a").i, 1);
  VCHECK_EQ(doc.map.at("b").str, "two");
}

VTEST(nested_map_one_level) {
  auto doc = parse("top: 1\nlimits:\n  a: 2\n  b: [3, 4, 5]\nafter: 6\n", "t");
  const Value& lim = doc.map.at("limits");
  VCHECK(lim.kind == Value::Kind::Map);
  VCHECK_EQ(lim.map.at("a").i, 2);
  VCHECK_EQ(lim.map.at("b").list.size(), size_t{3});
  VCHECK_EQ(lim.map.at("b").list[2].i, 5);
  VCHECK_EQ(doc.map.at("after").i, 6);
}

VTEST(list_of_ints) {
  auto doc = parse("dims: [1024, 1024, 64]\n", "t");
  VCHECK_EQ(doc.map.at("dims").list[0].i, 1024);
  VCHECK_EQ(doc.map.at("dims").list[2].i, 64);
}

VTEST(error_reports_line_number) {
  auto err = VCAPTURE(Error, parse("a: 1\nbroken line\n", "myfile"));
  VCHECK(err.code() == Err::ProfileParse);
  VCHECK_CONTAINS(err.what(), "myfile:2");
}

VTEST(error_duplicate_key) {
  auto err = VCAPTURE(Error, parse("a: 1\na: 2\n", "t"));
  VCHECK(err.code() == Err::ProfileParse);
  VCHECK_CONTAINS(err.what(), "duplicate key: a");
}

VTEST(error_deep_nesting_rejected) {
  auto err = VCAPTURE(Error, parse("a:\n  b:\n", "t"));
  VCHECK(err.code() == Err::ProfileParse);
}

VTEST(error_tab_indent_rejected) {
  auto err = VCAPTURE(Error, parse("a:\n\tb: 1\n", "t"));
  VCHECK(err.code() == Err::ProfileParse);
}

VTEST_MAIN
