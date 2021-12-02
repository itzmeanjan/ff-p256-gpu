#include <bench_ntt.hpp>

int64_t benchmark_six_step_fft(sycl::queue &q, const uint64_t dim,
                               const uint64_t wg_size, data_transfer_t choice) {
  ff_p254_t *vec_h =
      static_cast<ff_p254_t *>(sycl::malloc_host(sizeof(ff_p254_t) * dim, q));
  ff_p254_t *vec_d =
      static_cast<ff_p254_t *>(sycl::malloc_device(sizeof(ff_p254_t) * dim, q));

  prepare_random_vector(vec_h, dim);

  tp start, end;

  if (choice == data_transfer_t::both ||
      choice == data_transfer_t::only_host_to_device) {
    start = std::chrono::system_clock::now();
  }

  // host to device transfer
  q.memcpy(vec_d, vec_h, sizeof(ff_p254_t) * dim).wait();

  if (choice == data_transfer_t::none ||
      choice == data_transfer_t::only_device_to_host) {
    start = std::chrono::system_clock::now();
  }

  // compute
  six_step_fft(q, vec_d, dim, wg_size);

  if (choice == data_transfer_t::none ||
      choice == data_transfer_t::only_host_to_device) {
    end = std::chrono::system_clock::now();
  }

  // device to host transfer
  q.memcpy(vec_h, vec_d, sizeof(ff_p254_t) * dim).wait();

  if (choice == data_transfer_t::both ||
      choice == data_transfer_t::only_device_to_host) {
    end = std::chrono::system_clock::now();
  }

  sycl::free(vec_h, q);
  sycl::free(vec_d, q);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int64_t benchmark_six_step_ifft(sycl::queue &q, const uint64_t dim,
                                const uint64_t wg_size,
                                data_transfer_t choice) {
  ff_p254_t *vec_h =
      static_cast<ff_p254_t *>(sycl::malloc_host(sizeof(ff_p254_t) * dim, q));
  ff_p254_t *vec_d =
      static_cast<ff_p254_t *>(sycl::malloc_device(sizeof(ff_p254_t) * dim, q));

  prepare_random_vector(vec_h, dim);

  tp start, end;

  if (choice == data_transfer_t::both ||
      choice == data_transfer_t::only_host_to_device) {
    start = std::chrono::system_clock::now();
  }

  // host to device transfer
  q.memcpy(vec_d, vec_h, sizeof(ff_p254_t) * dim).wait();

  if (choice == data_transfer_t::none ||
      choice == data_transfer_t::only_device_to_host) {
    start = std::chrono::system_clock::now();
  }

  // compute
  six_step_ifft(q, vec_d, dim, wg_size);

  if (choice == data_transfer_t::none ||
      choice == data_transfer_t::only_host_to_device) {
    end = std::chrono::system_clock::now();
  }

  // device to host transfer
  q.memcpy(vec_h, vec_d, sizeof(ff_p254_t) * dim).wait();

  if (choice == data_transfer_t::both ||
      choice == data_transfer_t::only_device_to_host) {
    end = std::chrono::system_clock::now();
  }

  sycl::free(vec_h, q);
  sycl::free(vec_d, q);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}
