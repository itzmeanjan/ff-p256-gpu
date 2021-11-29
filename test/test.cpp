#include <test.hpp>

void test_two_adic_root_of_unity() {
  ff_p256_t one(1_ZL);
  ff_p256_t mod(mod_p256);
  ff_p256_t t(268435456_ZL); // 1 << 28
  ff_p256_t k = (mod - one) / t;

  assert(static_cast<ff_p256_t>(cbn::mod_exp(
             GENERATOR.data, k.data, mod_p256_bn)) == TWO_ADIC_ROOT_OF_UNITY);
}
