#include <bench_ntt.hpp>

int64_t benchmark_six_step_fft(sycl::queue &q, const uint64_t dim,
                               const uint64_t wg_size) {
  ff_p254_t *vec_h =
      static_cast<ff_p254_t *>(sycl::malloc_host(sizeof(ff_p254_t) * dim, q));
  ff_p254_t *vec_d =
      static_cast<ff_p254_t *>(sycl::malloc_device(sizeof(ff_p254_t) * dim, q));

  prepare_random_vector(vec_h, dim);
  q.memcpy(vec_d, vec_h, sizeof(ff_p254_t) * dim).wait();

  tp start = std::chrono::steady_clock::now();
  six_step_fft(q, vec_d, dim, wg_size);
  tp end = std::chrono::steady_clock::now();

  sycl::free(vec_h, q);
  sycl::free(vec_d, q);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}
