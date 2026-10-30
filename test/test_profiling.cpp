#include "include/test_neso_particles.hpp"

TEST(Profiling, per_particle_average) {

  std::size_t key1 = 1;
  std::size_t key2 = 2;

  auto r0 =
      ProfileRegion("key1", "key2", ProfileRegionType::PER_PARTICLE_TIMING);

  ASSERT_EQ(r0.profile_region_type_coeff, 0.0);

  r0.end(key1, key2, 1024);

  ASSERT_EQ(r0.profile_region_type_coeff, static_cast<double>(1024));

  ProfileMap profile_map;
  profile_map.enable();
  auto r1 = profile_map.start_region("key1", "key2",
                                     ProfileRegionType::PER_PARTICLE_TIMING);

  profile_map.end_region(r1, key1, key2, 1024);
  ASSERT_EQ(r1.value().profile_region_type_coeff, static_cast<double>(1024));
}
