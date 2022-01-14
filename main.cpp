#include <iostream>

#if defined TEST || defined BENCHMARK
#include <iomanip>
#endif

#if defined TEST
#include "test.hpp"
#elif defined BENCHMARK
#include "bench_ntt.hpp"
#endif

#if defined TEST || defined BENCHMARK
constexpr size_t WG_SIZE = 1 << 5;
#endif

int
main(int argc, char** argv)
{

#if defined TEST || defined BENCHMARK
  sycl::default_selector sel{};
  sycl::device d{ sel };
  sycl::context c{ d };
  sycl::queue q{ c, d, sycl::property::queue::enable_profiling{} };

  std::cout << "running on " << d.get_info<sycl::info::device::name>() << "\n"
            << std::endl;
#endif

#if defined TEST // only run test cases !

  test_two_adic_root_of_unity();
  test_get_root_of_unity();
  std::cout << "passed prime field tests !" << std::endl;

  // check with rectangular matrix
  test_matrix_transposed_initialise(q, 1ul << 15, WG_SIZE);
  std::cout << "passed matrix transposed initialisation tests ! [rectangular]"
            << std::endl;
  // check with square matrix
  test_matrix_transposed_initialise(q, 1ul << 16, WG_SIZE);
  std::cout << "passed matrix transposed initialisation tests ! [square]"
            << std::endl;

  // takes square matrix of dim x dim size, transposes twice
  // finally asserts with original matrix
  test_matrix_transpose(q, 1ul << 10, WG_SIZE);
  std::cout << "passed matrix transposition tests !" << std::endl;

  test_twiddle_multiplication(q, 1ul << 15, WG_SIZE);
  std::cout << "passed twiddle multiplication tests ! [rectangular]"
            << std::endl;
  test_twiddle_multiplication(q, 1ul << 16, WG_SIZE);
  std::cout << "passed twiddle multiplication tests ! [square]" << std::endl;

  test_six_step_fft_ifft(q, 1ul << 17, WG_SIZE);
  std::cout << "passed fft/ifft tests !" << std::endl;

#elif defined BENCHMARK // only run benchmarks !

  std::cout << "\nSix-Step FFT (without data transfer cost)\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    sycl::cl_ulong tm = benchmark_six_step_fft(q, 1ul << dim, WG_SIZE);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (double)tm * 1e-6 << " ms"
              << std::endl;
  }

  std::cout << "\nSix-Step IFFT (without data transfer cost)\n" << std::endl;
  std::cout << std::setw(11) << "dimension"
            << "\t\t" << std::setw(15) << "total" << std::endl;

  for (uint dim = 16; dim <= 23; dim++) {
    sycl::cl_ulong tm = benchmark_six_step_ifft(q, 1ul << dim, WG_SIZE);

    std::cout << std::setw(9) << std::right << (1ul << dim) << "\t\t"
              << std::setw(15) << std::right << (double)tm * 1e-6 << " ms"
              << std::endl;
  }

#else // do nothing useful !

#pragma message(                                                               \
  "Set **DO_RUN** variable to either `test` or `benchmark` when invoking make !")

  std::cout << "Check "
               "https://github.com/itzmeanjan/ff-p254-gpu/blob/"
               "288091435f7af607cf4681ebad54667effc4ecbc/Makefile#L8-L19"
            << std::endl;
#endif

  return 0;
}
