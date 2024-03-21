#ifndef _NESO_PARTICLES_SERIAL_INTERFACE_HPP_
#define _NESO_PARTICLES_SERIAL_INTERFACE_HPP_

#include <cstddef>
#include <cstdlib>

namespace NESO::Particles::MeshHierarchyData {

/**
 * Abstract base class for types to serialise into bytes.
 */
class SerialInterface {
public:
  /**
   * @returns The number of bytes required to serialise this instance.
   */
  virtual inline std::size_t get_num_bytes() const = 0;

  /**
   * Serialise this instance into the provided space.
   *
   * @param buffer[in, out] Pointer to space that the calling function
   * guarantees to be at least get_num_bytes in size.
   * @param num_bytes Size of allocated buffer passed (get_num_bytes).
   */
  virtual inline void serialise(std::byte *buffer,
                                const std::size_t num_bytes) const = 0;

  /**
   * @param buffer Pointer to space that the calling function guarantees to be
   * at least get_num_bytes in size from which this object should be recreated.
   * @param num_bytes Size of allocated buffer passed (get_num_bytes).
   */
  virtual inline void deserialise(const std::byte *buffer,
                                  const std::size_t num_bytes) = 0;
};

} // namespace NESO::Particles::MeshHierarchyData

#endif
