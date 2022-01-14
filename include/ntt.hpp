#pragma once
#include <CL/sycl.hpp>
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

// taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L3-L6
SYCL_EXTERNAL ff_p254_t
get_root_of_unity(uint64_t n);

// Initialises destination vector in transposed form of source vector
//
// Taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L544-L569
sycl::event
matrix_transposed_initialise(sycl::queue& q,
                             ff_p254_t* vec_src,
                             ff_p254_t* vec_dst,
                             const uint64_t rows,
                             const uint64_t cols,
                             const uint64_t width,
                             const uint64_t wg_size,
                             std::vector<sycl::event> evts);

// Parallel in-place square matrix transposition
//
// Taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L468-L542
sycl::event
matrix_transpose(sycl::queue& q,
                 ff_p254_t* data,
                 const uint64_t dim,
                 std::vector<sycl::event> evts);

// Exponentiates ω ( read N-th root of unity, when N = NTT domain size )
// to (r * c), where r = row index, c = column index
//
// Note this routine, has a dependency on `compute_twiddles` routine
// ( at
// https://github.com/itzmeanjan/ff-p254-gpu/blob/197821fc87d715f1cd1d11d5c8488b7a2f1f81ed/include/ntt.hpp#L49
// ) for twiddle factors i.e. ω raised upto R-th power, where R = row count of
// six step NTT matrix
//
// Taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L728-L751
sycl::event
twiddle_multiplication(sycl::queue& q,
                       ff_p254_t* vec,
                       ff_p254_t* omega,
                       const uint64_t rows,
                       const uint64_t cols,
                       const uint64_t width,
                       const uint64_t wg_size,
                       std::vector<sycl::event> evts);

// Performs `rows`-many parallel `cols`-point NTT
//
// Taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L571-L709
// and used in stripped down form
sycl::event
row_wise_transform(sycl::queue& q,
                   ff_p254_t* vec,
                   ff_p254_t* omega,
                   const uint64_t rows,
                   const uint64_t cols,
                   const uint64_t width,
                   const uint64_t wg_size,
                   std::vector<sycl::event> evts);

// Some utility function for index manipulation,
// taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L190-L217
SYCL_EXTERNAL uint64_t
bit_rev(uint64_t v, uint64_t max_bit_width);
SYCL_EXTERNAL uint64_t
rev_all_bits(uint64_t n);
SYCL_EXTERNAL uint64_t
permute_index(uint64_t idx, uint64_t size);

// Six step algorithm based NTT implementation for 254-bit prime field
//
// Taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L753-L831
sycl::event
six_step_fft(sycl::queue& q,
             ff_p254_t* vec,
             ff_p254_t* vec_scratch,
             ff_p254_t* omega_dim,
             ff_p254_t* omega_n1,
             ff_p254_t* omega_n2,
             const uint64_t dim,
             const uint64_t wg_size,
             std::vector<sycl::event> evts);

// Six step algorithm based NTT implementation for 254-bit prime field
//
// Taken from
// https://github.com/itzmeanjan/ff-gpu/blob/2f58f3d4a38d9f4a8db4f57faab352b1b16b9e0b/ntt.cpp#L833-L921
sycl::event
six_step_ifft(sycl::queue& q,
              ff_p254_t* vec,
              ff_p254_t* vec_scratch,
              ff_p254_t* omega_dim_inv,
              ff_p254_t* omega_n1_inv,
              ff_p254_t* omega_n2_inv,
              ff_p254_t* omega_domain_size_inv,
              const uint64_t dim,
              const uint64_t wg_size,
              std::vector<sycl::event> evts);
