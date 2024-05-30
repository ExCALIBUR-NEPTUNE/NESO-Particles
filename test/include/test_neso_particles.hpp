#ifndef _NESO_PARTICLES_TEST_NESO_PARTICLES_H_
#define _NESO_PARTICLES_TEST_NESO_PARTICLES_H_

#include "test_base.hpp"
#include "test_particle_group.hpp"
#include <cstdlib>
#include <filesystem>
#include <string>

#ifndef TO_STRING_MACRO
#define TO_STRING_MACRO(s) #s
#endif

namespace NESO::Particles {

template <typename T> inline auto as_orig_t(std::shared_ptr<T> ptr) {
  return std::dynamic_pointer_cast<typename TestMapperToNP<T>::type>(ptr);
}

template <typename T, typename... ARGS>
inline auto make_test_obj(ARGS... args) {
  return std::make_shared<typename NPToTestMapper<T>::type>(args...);
}

inline std::filesystem::path get_test_resource(std::string resource_name) {

  std::vector<std::filesystem::path> directories;
  directories.reserve(3);

  // Allow a pointing directly to test_resources with an environment variable
  char *env_test_resources = getenv("NESO_PARTICLES_TEST_RESOURCES_DIR");
  if (env_test_resources) {
    directories.push_back(std::filesystem::path(env_test_resources));
  }
  // If we are in <git root>/build then this points to
  // <git root>/tests/test_resources.
  directories.push_back(std::filesystem::path("../test/test_resources"));
  // If the test binary is in <install location>/bin then this points to
  // <install location>/test_resources
#ifdef CMAKE_INSTALL_PREFIX
  directories.push_back(
      std::filesystem::path(TO_STRING_MACRO(CMAKE_INSTALL_PREFIX)) /
      std::filesystem::path("test_resources"));
#endif

  const std::filesystem::path resource_base{resource_name};
  for (auto &root : directories) {
    const auto resource_path = std::filesystem::absolute(root / resource_base);
    if (std::filesystem::exists(resource_path)) {
      return resource_path;
    }
  }

  std::filesystem::path path;
  path.clear();
  return path;
}

#ifndef GET_TEST_RESOURCE
#define GET_TEST_RESOURCE(x, y)                                                \
  {                                                                            \
    x = get_test_resource(y);                                                  \
    if (x.empty()) {                                                           \
      GTEST_SKIP() << "Skipping test as resource missing: " << y;              \
    }                                                                          \
  }
#endif

} // namespace NESO::Particles

using namespace NESO::Particles;

#endif
