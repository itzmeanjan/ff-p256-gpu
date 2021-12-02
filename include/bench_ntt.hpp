#pragma once
#include <ntt.hpp>
#include <utils.hpp>

int64_t benchmark_six_step_fft(sycl::queue &q, const uint64_t dim,
                               const uint64_t wg_size);

int64_t benchmark_six_step_ifft(sycl::queue &q, const uint64_t dim,
                                const uint64_t wg_size);
