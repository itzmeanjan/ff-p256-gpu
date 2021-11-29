#include <iostream>
#include <test.hpp>

int main(int argc, char **argv) {
  sycl::device d{sycl::default_selector{}};
  sycl::queue q{d};

  test_two_adic_root_of_unity();
  test_get_root_of_unity();
  // check with rectangular matrix
  test_matrix_transposed_initialise(q, 1ul << 23, 1 << 7);
  // check with square matrix
  test_matrix_transposed_initialise(q, 1ul << 24, 1 << 7);

  std::cout << "passed all tests !" << std::endl;

  return 0;
}
