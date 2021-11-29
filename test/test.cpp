#include <test.hpp>

void test_two_adic_root_of_unity() {
  ff_p256_t one(1_ZL);
  ff_p256_t mod(mod_p256);
  ff_p256_t t(268435456_ZL); // 1 << 28
  ff_p256_t k = (mod - one) / t;

  assert(static_cast<ff_p256_t>(cbn::mod_exp(
             GENERATOR.data, k.data, mod_p256_bn)) == TWO_ADIC_ROOT_OF_UNITY);
}

void test_get_root_of_unity() {
  ff_p256_t one(1_ZL);
  ff_p256_t t_1(268435456_ZL); // 1 << 28
  ff_p256_t t_2(134217728_ZL); // 1 << 27
  ff_p256_t root_28 = get_root_of_unity(28);

  assert(root_28 == TWO_ADIC_ROOT_OF_UNITY);
  assert(static_cast<ff_p256_t>(
             cbn::mod_exp(root_28.data, t_1.data, mod_p256_bn)) == one);

  ff_p256_t root_27 = get_root_of_unity(27);
  ff_p256_t exp = static_cast<ff_p256_t>(
      cbn::mod_exp(root_28.data, cbn::to_big_int(2_ZL), mod_p256_bn));

  assert(root_27 == exp);
  assert(static_cast<ff_p256_t>(
             cbn::mod_exp(root_27.data, t_2.data, mod_p256_bn)) == one);
}
