#include <iostream>
#include <test.hpp>

int main(int argc, char **argv) {
  sycl::device d{sycl::default_selector{}};
  sycl::queue q{d};
  std::cout << "running on " << d.get_info<sycl::info::device::name>() << "\n"
            << std::endl;

  test_two_adic_root_of_unity();
  test_get_root_of_unity();
  std::cout << "passed prime field tests !" << std::endl;

  // check with rectangular matrix
  test_matrix_transposed_initialise(q, 1ul << 23, 1ul << 7);
  // check with square matrix
  test_matrix_transposed_initialise(q, 1ul << 24, 1ul << 7);
  std::cout << "passed matrix transposed initialisation tests !" << std::endl;

  // takes square matrix of dim x dim size, transposes twice
  // finally asserts with original matrix
  test_matrix_transpose(q, 1ul << 10, 1ul << 6);
  std::cout << "passed matrix transposition tests !" << std::endl;

  test_compute_twiddles(q, 1ul << 23, 1ul << 7);
  std::cout << "passed twiddle compute tests !" << std::endl;

  return 0;
}
