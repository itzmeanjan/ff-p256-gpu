#include <ntt.hpp>

ff_p256_t get_root_of_unity(uint64_t n) {
  uint64_t pow_ = 1ul << (28 - n);
  ff_p256_t pow(pow_);

  return static_cast<ff_p256_t>(
      cbn::mod_exp(TWO_ADIC_ROOT_OF_UNITY.data, pow.data, mod_p256_bn));
}

sycl::event matrix_transposed_initialise(
    sycl::queue &q, ff_p256_t *vec_src, ff_p256_t *vec_dst, const uint64_t rows,
    const uint64_t cols, const uint64_t width, const uint64_t wg_size,
    std::vector<sycl::event> evts) {
  return q.submit([&](sycl::handler &h) {
    h.depends_on(evts);

    h.parallel_for<class kernelMatrixTransposedInitialise>(
        sycl::nd_range<2>{sycl::range<2>{rows, cols},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          sycl::sub_group sg = it.get_sub_group();

          const size_t r = it.get_global_id(0);
          const size_t c = it.get_global_id(1);

          const uint64_t width_ = sycl::group_broadcast(sg, width);

          *(vec_dst + r * width_ + c) = *(vec_src + c * width_ + r);
        });
  });
}
