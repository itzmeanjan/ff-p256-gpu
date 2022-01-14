#pragma once
#include <ctbignum/ctbignum.hpp>

using namespace cbn::literals;

constexpr auto mod_p254 =
  21888242871839275222246405745257275088548364400416034343698204186575808495617_ZL;
constexpr auto mod_p254_bn = cbn::to_big_int(mod_p254);

using ff_p254_t = decltype(cbn::Zq(mod_p254));

// primitive element of prime field
constexpr ff_p254_t GENERATOR(5_ZL);

// assert ((mod_p254 - 1) >> 28) & 0b1 == 1
const uint64_t TWO_ADICITY_ = 28ul;
constexpr ff_p254_t TWO_ADICITY(28_ZL);

// generator ** ((mod_p254 - 1) >> 28)
constexpr ff_p254_t TWO_ADIC_ROOT_OF_UNITY(
  19103219067921713944291392827692070036145651957329286315305642004821462161904_ZL);
