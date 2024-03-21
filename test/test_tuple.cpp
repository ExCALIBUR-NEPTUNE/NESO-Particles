#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <type_traits>

using namespace NESO::Particles;

TEST(Tuple, tuple) {
  using Tuple0 = Tuple::Tuple<int, int64_t, double>;
  static_assert(std::is_trivially_copyable<Tuple0>::value == true);
  static_assert(std::is_same<Tuple::GetIndexType<0, int, int64_t, double>::type,
                             int>::value == true);
  static_assert(std::is_same<Tuple::GetIndexType<1, int, int64_t, double>::type,
                             int64_t>::value == true);
  static_assert(std::is_same<Tuple::GetIndexType<2, int, int64_t, double>::type,
                             double>::value == true);

  Tuple0 t;

  auto t0 = static_cast<Tuple::TupleImpl<0, int> *>(&t);
  t0->value = 42;
  auto t1 = static_cast<Tuple::TupleImpl<2, double> *>(&t);
  t1->value = 3.14;

  auto to_test_0 = static_cast<Tuple::TupleImpl<0, int> *>(&t)->value;
  auto to_test_1 = static_cast<Tuple::TupleImpl<2, double> *>(&t)->value;
  EXPECT_EQ(to_test_0, 42);
  EXPECT_EQ(to_test_1, 3.14);

  Tuple::get<0>(t) = 43;
  Tuple::get<2>(t) = 3.141;

  EXPECT_EQ(Tuple::get<0>(t), 43);
  EXPECT_EQ(Tuple::get<2>(t), 3.141);
}

TEST(Tuple, apply) {
  using Tuple0 = Tuple::Tuple<int, int64_t, double>;
  Tuple0 t;
  Tuple::get<0>(t) = -42;
  Tuple::get<1>(t) = 43;
  Tuple::get<2>(t) = 3.141;

  int aa;
  int64_t bb;
  double cc;

  const int to_test = Tuple::apply(
      [&](const int a, const int64_t b, const double c) {
        aa = a;
        bb = b;
        cc = c;
        return 53;
      },
      t);

  EXPECT_EQ(to_test, 53);
  EXPECT_EQ(Tuple::get<0>(t), aa);
  EXPECT_EQ(Tuple::get<1>(t), bb);
  EXPECT_EQ(Tuple::get<2>(t), cc);
}
