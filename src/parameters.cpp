#include <neso_particles/parameters.hpp>

namespace NESO::Particles {

std::size_t Parameters::get_env_size_t(const std::string name,
                                       const std::size_t default_value) {

  if (this->contains(name)) {
    const std::shared_ptr<SizeTParameter> value_ptr =
        this->get<SizeTParameter>(name);
    NESOASSERT(value_ptr != nullptr, "Bad key-value pair.");
    return value_ptr->value;
  } else {

    const std::size_t value =
        NESO::Particles::get_env_size_t(name, default_value);
    auto value_ptr = std::make_shared<SizeTParameter>(value);
    this->set<SizeTParameter>(name, value_ptr);
    return value;
  }
}
} // namespace NESO::Particles
