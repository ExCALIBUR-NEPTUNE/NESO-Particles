#ifndef _NESO_PARTICLES_SERIAL_CONTAINER_HPP_
#define _NESO_PARTICLES_SERIAL_CONTAINER_HPP_

#include "../typedefs.hpp"
#include "serial_interface.hpp"
#include <cstring>
#include <limits>
#include <vector>

namespace NESO::Particles::MeshHierarchyData {

/**
 * TODO
 */
template <typename T> class SerialContainer {

public:
  /// TODO
  std::vector<std::byte> buffer;
  SerialContainer() = default;

  /**
   * TODO
   */
  SerialContainer(const std::size_t num_bytes) {
    this->buffer.resize(num_bytes);
  }

  /**
   * TODO
   */
  SerialContainer(const std::vector<T> &inputs) {
    const int num_inputs = inputs.size();
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> sizes;
    offsets.reserve(num_inputs);
    sizes.reserve(num_inputs);

    // Get the size of each object to pass to allocate space and compute offsets
    std::size_t offset = 0;
    for (auto &ix : inputs) {
      const std::size_t sizex = ix.get_num_bytes();
      sizes.push_back(sizex);
      offsets.push_back(offset);
      offset += sizex + sizeof(std::size_t);
    }
    this->buffer.resize(offset);

    // Serialise the objects into the buffer.
    std::byte *buffer_ptr = this->buffer.data();
    for (int ix = 0; ix < num_inputs; ix++) {
      std::size_t size = sizes.at(ix);
      std::memcpy(buffer_ptr, &size, sizeof(std::size_t));
      buffer_ptr += sizeof(std::size_t);
      inputs.at(ix).serialise(buffer_ptr, size);
      buffer_ptr += size;
    }

    NESOASSERT(buffer_ptr == this->buffer.data() + offset,
               "Miss-match between offset and buffer size.");
  }

  /**
   * TODO
   */
  inline void get(std::vector<T> &outputs) const {
    const size_t num_bytes = this->buffer.size();
    outputs.clear();
    if (num_bytes == 0) {
      return;
    }

    // Determine how many outputs there are.
    const std::byte *buffer_ptr = this->buffer.data();
    const std::byte *buffer_end = buffer_ptr + num_bytes;
    std::size_t num_outputs = 0;
    while (buffer_ptr < buffer_end) {
      std::size_t size = std::numeric_limits<std::size_t>::max();
      std::memcpy(&size, buffer_ptr, sizeof(std::size_t));
      NESOASSERT(size < std::numeric_limits<std::size_t>::max(),
                 "Bad value unpacked.");
      buffer_ptr += size + sizeof(std::size_t);
      num_outputs++;
    }
    // Deserialise the objects into the output vector.
    outputs.resize(num_outputs);
    buffer_ptr = this->buffer.data();
    auto outputs_ptr = outputs.data();
    while (buffer_ptr < buffer_end) {
      std::size_t size = -1;
      std::memcpy(&size, buffer_ptr, sizeof(std::size_t));
      buffer_ptr += sizeof(std::size_t);
      outputs_ptr->deserialise(buffer_ptr, size);
      outputs_ptr++;
      buffer_ptr += size;
    }
  }

  /**
   * TODO
   */
  inline void append(SerialContainer<T> &other) {
    this->buffer.reserve(this->buffer.size() + other.buffer.size());
    this->buffer.insert(this->buffer.end(), other.buffer.begin(),
                        other.buffer.end());
  }
};

} // namespace NESO::Particles::MeshHierarchyData

#endif
