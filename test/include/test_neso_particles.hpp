#ifndef _NESO_PARTICLES_TEST_NESO_PARTICLES_H_
#define _NESO_PARTICLES_TEST_NESO_PARTICLES_H_

#include "test_base.hpp"
#include "test_particle_group.hpp"

namespace NESO::Particles {

template <typename T> inline auto as_orig_t(std::shared_ptr<T> ptr) {
  return std::dynamic_pointer_cast<typename TestMapperToNP<T>::type>(ptr);
}

template <typename T, typename... ARGS>
inline auto make_test_obj(ARGS... args) {
  return std::make_shared<typename NPToTestMapper<T>::type>(args...);
}

} // namespace NESO::Particles

using namespace NESO::Particles;

#endif
