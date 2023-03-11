#pragma once

#include <cstdint>
#include <cassert>

struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

rgba8_t heat_lut(float x);