#pragma once
#include "field.hpp"
#include <chrono>
#include <random>

typedef std::chrono::_V2::system_clock::time_point tp;

void
prepare_random_vector(ff_p254_t* vec, uint64_t n)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(1ul, 1ul << 63);

  for (uint64_t i = 0; i < n; i++) {
    *(vec + i) = ff_p254_t(dis(gen));
  }
}
