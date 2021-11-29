#pragma once
#include <CL/sycl.hpp>
#include <ctbignum/ctbignum.hpp>

using namespace cbn::literals;

constexpr auto mod_p256 =
    21888242871839275222246405745257275088548364400416034343698204186575808495617_ZL;
constexpr auto mod_p256_bn = cbn::to_big_int(mod_p256);

using ff_p256_t = decltype(cbn::Zq(mod_p256));

// primitive element of prime field
constexpr ff_p256_t GENERATOR(5_ZL);

// assert ((mod_p256 - 1) >> 28) & 0b1 == 1
const uint64_t  TWO_ADICITY_ = 28ul;
constexpr ff_p256_t TWO_ADICITY(28_ZL);

// generator ** ((mod_p256 - 1) >> 28)
constexpr ff_p256_t TWO_ADIC_ROOT_OF_UNITY(
    19103219067921713944291392827692070036145651957329286315305642004821462161904_ZL);

// taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L3-L6
ff_p256_t get_root_of_unity(uint64_t n);

// Initialises destination vector in transposed form of source vector
//
// Taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L544-L569
sycl::event matrix_transposed_initialise(
    sycl::queue &q, ff_p256_t *vec_src, ff_p256_t *vec_dst, const uint64_t rows,
    const uint64_t cols, const uint64_t width, const uint64_t wg_size,
    std::vector<sycl::event> evts);
