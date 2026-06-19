#ifndef _NESO_PARTICLES_MESH_HIERARCHY_DATA_GENERIC_SERIAL_CONTAINER_HPP_
#define _NESO_PARTICLES_MESH_HIERARCHY_DATA_GENERIC_SERIAL_CONTAINER_HPP_

#include "serial_interface.hpp"
#include <cstring>

namespace NESO::Particles::MeshHierarchyData {

/**
 * Generic serial container for a copyable type. The wrapped type, T, should be
 * trivially copyable and default constructable.
 */
template <typename T> struct GenericSerialContainer : public SerialInterface {
  // The wrapped object.
  T obj;

  /**
   * Default constructor.
   */
  GenericSerialContainer() = default;

  /**
   * Create container around an object.
   */
  GenericSerialContainer(const T &a) : obj(a) {}

  virtual ~GenericSerialContainer() = default;
  virtual inline std::size_t get_num_bytes() const override {
    return sizeof(T);
  }
  virtual inline void
  serialise([[maybe_unused]] std::byte *buffer,
            [[maybe_unused]] const std::size_t num_bytes) const override {
    std::memcpy(buffer, &this->obj, sizeof(T));
  }
  virtual inline void
  deserialise([[maybe_unused]] const std::byte *buffer,
              [[maybe_unused]] const std::size_t num_bytes) override {
    std::memcpy(&this->obj, buffer, sizeof(T));
  }
};
} // namespace NESO::Particles::MeshHierarchyData

#endif
