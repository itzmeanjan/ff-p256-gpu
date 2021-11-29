#include <ntt.hpp>

ff_p256_t get_root_of_unity(uint64_t n) {
  uint64_t pow_ = 1ul << (28 - n);
  ff_p256_t pow(pow_);

  return static_cast<ff_p256_t>(
      cbn::mod_exp(TWO_ADIC_ROOT_OF_UNITY.data, pow.data, mod_p256_bn));
}
