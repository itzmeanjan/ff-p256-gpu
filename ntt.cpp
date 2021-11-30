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

sycl::event matrix_transpose(sycl::queue &q, ff_p256_t *data,
                             const uint64_t dim,
                             std::vector<sycl::event> evts) {
  constexpr size_t TILE_DIM = 1 << 4;
  constexpr size_t BLOCK_ROWS = 1 << 3;

  assert(TILE_DIM >= BLOCK_ROWS);

  return q.submit([&](sycl::handler &h) {
    sycl::accessor<ff_p256_t, 2, sycl::access_mode::read_write,
                   sycl::target::local>
        tile_s{sycl::range<2>{TILE_DIM, TILE_DIM + 1}, h};
    sycl::accessor<ff_p256_t, 2, sycl::access_mode::read_write,
                   sycl::target::local>
        tile_d{sycl::range<2>{TILE_DIM, TILE_DIM + 1}, h};

    h.depends_on(evts);
    h.parallel_for<class kernelMatrixTransposition>(
        sycl::nd_range<2>{sycl::range<2>{dim / (TILE_DIM / BLOCK_ROWS), dim},
                          sycl::range<2>{BLOCK_ROWS, TILE_DIM}},
        [=](sycl::nd_item<2> it) {
          sycl::group<2> grp = it.get_group();
          const size_t grp_id_x = it.get_group().get_id(1);
          const size_t grp_id_y = it.get_group().get_id(0);
          const size_t loc_id_x = it.get_local_id(1);
          const size_t loc_id_y = it.get_local_id(0);
          const size_t grp_width_x = it.get_group().get_group_range(1);

          // @note x denotes index along x-axis
          // while y denotes index along y-axis
          //
          // so in usual (row, col) indexing of 2D array
          // row = y, col = x
          const size_t x = grp_id_x * TILE_DIM + loc_id_x;
          const size_t y = grp_id_y * TILE_DIM + loc_id_y;

          const size_t width = grp_width_x * TILE_DIM;

          // non-diagonal cell blocks
          if (grp_id_y > grp_id_x) {
            size_t dx = grp_id_y * TILE_DIM + loc_id_x;
            size_t dy = grp_id_x * TILE_DIM + loc_id_y;

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              tile_s[loc_id_y + j][loc_id_x] = *(data + (y + j) * width + x);
            }

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              tile_d[loc_id_y + j][loc_id_x] = *(data + (dy + j) * width + dx);
            }

            sycl::group_barrier(grp, sycl::memory_scope::work_group);

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              *(data + (dy + j) * width + dx) = tile_s[loc_id_x][loc_id_y + j];
            }

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              *(data + (y + j) * width + x) = tile_d[loc_id_x][loc_id_y + j];
            }

            return;
          }

          // diagonal cell blocks
          if (grp_id_y == grp_id_x) {
            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              tile_s[loc_id_y + j][loc_id_x] = *(data + (y + j) * width + x);
            }

            sycl::group_barrier(grp, sycl::memory_scope::work_group);

            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
              *(data + (y + j) * width + x) = tile_s[loc_id_x][loc_id_y + j];
            }
          }
        });
  });
}

sycl::event compute_twiddles(sycl::queue &q, ff_p256_t *twiddles,
                             ff_p256_t *omega, const uint64_t dim,
                             const uint64_t wg_size,
                             std::vector<sycl::event> evts) {
  return q.submit([&](sycl::handler &h) {
    sycl::accessor<ff_p256_t, 1, sycl::access_mode::read_write,
                   sycl::target::local>
        lds{sycl::range<1>{1}, h};

    h.depends_on(evts);
    h.parallel_for<class kernelComputeTwiddles>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          sycl::group<1> grp = it.get_group();

          // only work-group leader reads from global memory
          // and caches in local memory
          if (sycl::ext::oneapi::leader(grp)) {
            lds[0] = *omega;
          }

          // wait until all work-items of work-group reach here
          sycl::group_barrier(grp);

          const uint64_t c = it.get_global_id(0);

          // now all work-items of some work-group read
          // same (cached) ω from local memory
          *(twiddles + c) = static_cast<ff_p256_t>(
              cbn::mod_exp(lds[0].data, ff_p256_t(c).data, mod_p256_bn));
        });
  });
}

sycl::event twiddle_multiplication(sycl::queue &q, ff_p256_t *vec,
                                   ff_p256_t *twiddles, const uint64_t rows,
                                   const uint64_t cols, const uint64_t width,
                                   const uint64_t wg_size,
                                   std::vector<sycl::event> evts) {
  assert(cols == width || 2 * cols == width);

  return q.submit([&](sycl::handler &h) {
    h.depends_on(evts);
    h.parallel_for<class kernelTwiddleMultiplication>(
        sycl::nd_range<2>{sycl::range<2>{rows, cols},
                          sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint64_t r = it.get_global_id(0);
          const uint64_t c = it.get_global_id(1);

          *(vec + r * width + c) =
              *(vec + r * width + c) *
              static_cast<ff_p256_t>(cbn::mod_exp(
                  (*(twiddles + r)).data, ff_p256_t(c).data, mod_p256_bn));
        });
  });
}
