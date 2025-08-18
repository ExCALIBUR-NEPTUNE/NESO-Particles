#ifndef _NESO_PARTICLES_PARAMETERS_HPP_
#define _NESO_PARTICLES_PARAMETERS_HPP_

#include "typedefs.hpp"
#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace NESO::Particles {

/**
 * ABC for parameters to inherit.
 */
struct Parameter {
  virtual inline ~Parameter() = default;
};

/**
 * Type to store a vale of size_t.
 */
struct SizeTParameter : Parameter {
  std::size_t value;
  virtual ~SizeTParameter() = default;
  SizeTParameter() = default;
  SizeTParameter(const std::size_t value) : Parameter(), value(value) {}
};

/**
 * Generic Key, value store for paramters.
 */
struct Parameters {
  /// The key, value map for stored parameters.
  std::map<std::string, std::shared_ptr<Parameter>> values;

  /**
   * Retrive a value from the map.
   *
   * @param name Key for value to retrive.
   * @returns Value cast to the templated type.
   */
  template <typename T> inline std::shared_ptr<T> get(const std::string name) {
    auto v = std::dynamic_pointer_cast<T>(this->values.at(name));
    NESOASSERT(v != nullptr, "Could not get parameter with name: " + name);
    return v;
  }

  /**
   * Set a key, value pair.
   *
   * @param name Key to set with value.
   * @param value Value as a shared pointer to a type descendant from Parameter.
   */
  template <typename T>
  inline void set(const std::string name, std::shared_ptr<T> value) {
    auto v = std::dynamic_pointer_cast<Parameter>(value);
    NESOASSERT(v != nullptr, "Could not set parameter with name: " + name);
    this->values[name] = v;
  }

  /**
   * Test if key exists.
   * @param name Key to test existance of.
   * @returns True if key exists in the map.
   */
  inline bool contains(const std::string name) {
    return static_cast<bool>(this->values.count(name));
  }
};

} // namespace NESO::Particles

#endif
