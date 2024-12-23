#ifndef _NESO_PARTICLES_DMPLEX_CELL_SERIALISE_HPP_
#define _NESO_PARTICLES_DMPLEX_CELL_SERIALISE_HPP_

#include "../../mesh_hierarchy_data/serial_interface.hpp"
#include "petsc_common.hpp"
#include <cstring>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace NESO::Particles::PetscInterface {

/**
 * Representation of a DMPlex cell using std::map and std::vector.
 */
class CellSTDRepresentation {
protected:
  std::map<PetscInt, PetscInt> map_to_depth;

  /**
   * Populates the map_per_height
   */
  inline void traverse_cell_specification(
      DM &dm, PetscInt index,
      std::map<PetscInt, std::set<PetscInt>> &map_per_height) {
    PetscInt cone_size, height, depth, point_start, point_end;
    PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));
    NESOASSERT((index >= point_start) && (index < point_end),
               "Input point not in point range.");
    PETSCCHK(DMPlexGetDepth(dm, &depth));
    PETSCCHK(DMPlexGetPointHeight(dm, index, &height));
    NESOASSERT((height >= 0) && (height <= depth),
               "Bad point height returned.");

    map_per_height[height].insert(index);

    PETSCCHK(DMPlexGetConeSize(dm, index, &cone_size));
    const PetscInt *cone;
    PETSCCHK(DMPlexGetCone(dm, index, &cone));

    for (int cx = 0; cx < cone_size; cx++) {
      const PetscInt child_index = cone[cx];
      traverse_cell_specification(dm, child_index, map_per_height);
    }
  }

public:
  /// Map from a point to the points that form the cone for the point.
  std::map<PetscInt, std::vector<PetscInt>> point_specs;
  /// Map from points which are vertices to the coordinates.
  std::map<PetscInt, std::vector<PetscScalar>> vertices;

  CellSTDRepresentation() = default;

  /**
   * Construct a representation from a DMPlex and a petsc point index.
   *
   * @param dm DMPlex containing the cell to represent.
   * @param index PETSc point index in the DMPlex.
   * @param rename_function Function which maps local PETSc point indices to
   * global point indices.
   */
  CellSTDRepresentation(DM &dm, PetscInt index,
                        std::function<PetscInt(PetscInt)> &rename_function) {
    PetscInt depth, height;
    // Cells have height 0
    PETSCCHK(DMPlexGetPointHeight(dm, index, &height));
    PETSCCHK(DMPlexGetDepth(dm, &depth));

    std::map<PetscInt, std::set<PetscInt>> map_per_height;
    traverse_cell_specification(dm, index, map_per_height);

    for (auto hx : map_per_height) {
      for (auto ix : hx.second) {
        const PetscInt point_rename = rename_function(ix);
        PetscInt cone_size;
        PETSCCHK(DMPlexGetConeSize(dm, ix, &cone_size));
        const PetscInt *cone;
        PETSCCHK(DMPlexGetCone(dm, ix, &cone));
        this->point_specs[point_rename].reserve(cone_size);
        for (PetscInt px = 0; px < cone_size; px++) {
          this->point_specs[point_rename].push_back(rename_function(cone[px]));
        }
      }
    }

    for (auto vx : map_per_height[depth]) {
      PetscBool is_dg;
      PetscInt nc;
      const PetscScalar *array;
      PetscScalar *coords;
      PETSCCHK(DMPlexGetCellCoordinates(dm, vx, &is_dg, &nc, &array, &coords));
      const PetscInt global_index = rename_function(vx);
      this->vertices[global_index].reserve(nc);
      for (int cx = 0; cx < nc; cx++) {
        this->vertices[global_index].push_back(coords[cx]);
      }
      PETSCCHK(
          DMPlexRestoreCellCoordinates(dm, vx, &is_dg, &nc, &array, &coords));
    }
  }

  /**
   * For a point in the representation get the point depth.
   *
   * @param point Point to determine depth of.
   * @returns Depth of point.
   */
  inline PetscInt get_point_depth(const PetscInt point) {
    NESOASSERT(point_specs.count(point), "Unknown point index.");
    if (this->map_to_depth.count(point)) {
      return this->map_to_depth.at(point);
    } else if (point_specs.at(point).size()) {
      const PetscInt d = this->get_point_depth(point_specs.at(point).at(0)) + 1;
      this->map_to_depth[point] = d;
      return d;
    } else {
      // Is a vertex hence has depth 0
      return 0;
    }
  }

  /**
   * Get the depth of the stored representation.
   *
   * @returns Depth of stored graph.
   */
  inline PetscInt get_depth() {
    PetscInt d_max = std::numeric_limits<PetscInt>::lowest();
    PetscInt d_min = std::numeric_limits<PetscInt>::max();
    for (const auto &point : this->point_specs) {
      const auto d = this->get_point_depth(point.first);
      d_max = std::max(d_max, d);
      d_min = std::min(d_min, d);
    }
    NESOASSERT(d_min == 0, "Expected the minumum to be 0.");
    return d_max;
  }

  /**
   * Print the stored representation to stdout.
   */
  inline void print() {
    nprint("specification:");
    for (auto hx : point_specs) {
      nprint(std::to_string(hx.first) + ":");
      for (auto px : hx.second) {
        nprint("  ", px);
      }
    }
    for (auto vx : vertices) {
      nprint(std::to_string(vx.first) + ":");
      std::cout << "  ";
      for (auto cx : vx.second) {
        std::cout << cx << " ";
      }
      std::cout << std::endl;
    }
  }

  /**
   * Serialise the stored representation into the provided buffer.
   *
   * @param[in, out] Buffer to serialise the stored representation into.
   */
  inline void serialise(std::vector<std::byte> &buffer) {
    buffer.clear();
    std::vector<PetscInt> buffer_int;
    std::vector<PetscScalar> buffer_scalar;

    // push the point specifications
    buffer_int.push_back(this->point_specs.size());
    for (auto point_spec : this->point_specs) {
      buffer_int.push_back(point_spec.first);
      buffer_int.push_back(point_spec.second.size());
      for (auto spec : point_spec.second) {
        buffer_int.push_back(spec);
      }
    }

    // push the vertex coordinates
    buffer_int.push_back(this->vertices.size());
    for (auto px : this->vertices) {
      buffer_int.push_back(px.first);
      buffer_int.push_back(px.second.size());
      for (auto cx : px.second) {
        buffer_scalar.push_back(cx);
      }
    }

    // copy int and scalar data into the buffer
    const std::size_t num_bytes_meta = 2 * sizeof(std::size_t);
    const std::size_t num_bytes_int = buffer_int.size() * sizeof(PetscInt);
    const std::size_t num_bytes_scalar =
        buffer_scalar.size() * sizeof(PetscScalar);
    buffer.resize(num_bytes_meta + num_bytes_int + num_bytes_scalar);

    std::memcpy(buffer.data(), &num_bytes_int, sizeof(std::size_t));
    std::memcpy(buffer.data() + sizeof(std::size_t), &num_bytes_scalar,
                sizeof(std::size_t));
    std::memcpy(buffer.data() + num_bytes_meta, buffer_int.data(),
                num_bytes_int);
    std::memcpy(buffer.data() + num_bytes_meta + num_bytes_int,
                buffer_scalar.data(), num_bytes_scalar);
  }

  /**
   * Populate this struct instance with the serialised representation provided.
   *
   * @param buffer Serialised representation from which to populate the members
   * of the instance.
   */
  inline void deserialise(const std::vector<std::byte> &buffer) {
    const std::size_t num_bytes_meta = 2 * sizeof(std::size_t);
    NESOASSERT(buffer.size() >= num_bytes_meta,
               "Buffer is too small to hold meta data.");
    std::size_t num_bytes_int;
    std::size_t num_bytes_scalar;
    std::memcpy(&num_bytes_int, buffer.data(), sizeof(std::size_t));
    std::memcpy(&num_bytes_scalar, buffer.data() + sizeof(std::size_t),
                sizeof(std::size_t));

    const std::size_t num_int = num_bytes_int / sizeof(PetscInt);
    const std::size_t num_scalar = num_bytes_scalar / sizeof(PetscScalar);

    // This copy is probably avoidable with pointer type casts
    std::vector<PetscInt> buffer_int(num_int);
    std::vector<PetscScalar> buffer_scalar(num_scalar);

    std::memcpy(buffer_int.data(), buffer.data() + num_bytes_meta,
                num_bytes_int);
    std::memcpy(buffer_scalar.data(),
                buffer.data() + num_bytes_meta + num_bytes_int,
                num_bytes_scalar);

    size_t index = 0;
    const PetscInt num_points = buffer_int.at(index++);
    for (int px = 0; px < num_points; px++) {
      const PetscInt point = buffer_int.at(index++);
      const PetscInt cone_size = buffer_int.at(index++);
      std::vector<PetscInt> cone_elements;
      cone_elements.reserve(cone_size);
      for (int cx = 0; cx < cone_size; cx++) {
        const PetscInt child_index = buffer_int.at(index++);
        cone_elements.push_back(child_index);
      }
      this->point_specs[point] = cone_elements;
    }

    size_t index_scalar = 0;
    const PetscInt num_vertices = buffer_int.at(index++);
    for (int vx = 0; vx < num_vertices; vx++) {
      const PetscInt point = buffer_int.at(index++);
      const PetscInt num_coords = buffer_int.at(index++);
      this->vertices[point].reserve(num_coords);
      for (int cx = 0; cx < num_coords; cx++) {
        this->vertices.at(point).push_back(buffer_scalar.at(index_scalar++));
      }
    }

    NESOASSERT(index == num_int, "Error unpacking ints.");
    NESOASSERT(index_scalar == num_scalar, "Error unpacking scalars.");
  }
};

/**
 * Class to hold a serialised DMPlex cell along with metadata such as the
 * owning rank and local id on the owning rank. This class inherits the @ref
 * SerialInterface class such that the instances can be communicated and
 * combined.
 */
class DMPlexCellSerialise : public MeshHierarchyData::SerialInterface {
public:
  PetscInt cell_local_id;
  PetscInt cell_global_id;
  int owning_rank;
  DMPolytopeType cell_type;
  std::vector<std::byte> cell_representation;

  virtual ~DMPlexCellSerialise() = default;
  DMPlexCellSerialise() = default;

  /**
   * Create instance.
   *
   * @param cell_local_id Index of the cell on the owning MPI rank.
   * @param cell_global_id Global index of the cell in the DMPlex.
   * @param owning_rank MPI rank which owns the cell.
   * @param cell_type Type of the cell.
   * @param cell_representation Serialised representation on the cell.
   */
  DMPlexCellSerialise(PetscInt cell_local_id, PetscInt cell_global_id,
                      int owning_rank, DMPolytopeType cell_type,
                      std::vector<std::byte> &cell_representation)
      : cell_local_id(cell_local_id), cell_global_id(cell_global_id),
        owning_rank(owning_rank), cell_type(cell_type),
        cell_representation(cell_representation) {}

  /**
   * @returns The number of bytes required to store this instance when
   * serialised.
   */
  virtual inline std::size_t get_num_bytes() const override {
    return 2 * sizeof(PetscInt) + 2 * sizeof(int) +
           this->cell_representation.size();
  }

  /**
   * Serialise the representation into a buffer.
   *
   * @param[in, out] buffer Buffer to serialise into.
   * @param[in] num_bytes Size of buffer to perform validity checks against.
   */
  virtual inline void serialise(std::byte *buffer,
                                const std::size_t num_bytes) const override {
    std::byte *buffer_end = buffer + num_bytes;
    std::memcpy(buffer, &this->cell_local_id, sizeof(PetscInt));
    buffer += sizeof(PetscInt);
    std::memcpy(buffer, &this->cell_global_id, sizeof(PetscInt));
    buffer += sizeof(PetscInt);
    std::memcpy(buffer, &this->owning_rank, sizeof(int));
    buffer += sizeof(int);
    int cell_type_int = static_cast<int>(this->cell_type);
    std::memcpy(buffer, &cell_type_int, sizeof(int));
    buffer += sizeof(int);
    std::memcpy(buffer, this->cell_representation.data(),
                this->cell_representation.size());
    buffer += this->cell_representation.size();

    NESOASSERT(buffer == buffer_end, "Error packing cell.");
  }

  /**
   * Reconstruct the instance from the serialised form.
   *
   * @param buffer Input buffer with serial representation.
   * @param num_bytes Size of input buffer for reference and checking.
   */
  virtual inline void deserialise(const std::byte *buffer,
                                  const std::size_t num_bytes) override {
    auto buffer_end = buffer + num_bytes;
    std::memcpy(&this->cell_local_id, buffer, sizeof(PetscInt));
    buffer += sizeof(PetscInt);
    std::memcpy(&this->cell_global_id, buffer, sizeof(PetscInt));
    buffer += sizeof(PetscInt);
    std::memcpy(&this->owning_rank, buffer, sizeof(int));
    buffer += sizeof(int);
    int cell_type_int;
    std::memcpy(&cell_type_int, buffer, sizeof(int));
    buffer += sizeof(int);
    this->cell_type = static_cast<DMPolytopeType>(cell_type_int);
    const size_t num_bytes_remaining = buffer_end - buffer;
    this->cell_representation.resize(num_bytes_remaining);
    std::memcpy(this->cell_representation.data(), buffer, num_bytes_remaining);
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
