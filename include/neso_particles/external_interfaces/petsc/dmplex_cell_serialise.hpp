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
  void traverse_cell_specification(
      DM &dm, PetscInt index,
      std::map<PetscInt, std::set<PetscInt>> &map_per_height);

public:
  /// Map from a point to the points that form the cone for the point.
  std::map<PetscInt, std::vector<PetscInt>> point_specs;
  /// Map from a point to the orientations for each point in the cone.
  std::map<PetscInt, std::vector<PetscInt>> cone_orientations;
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
                        std::function<PetscInt(PetscInt)> &rename_function);

  /**
   * For a point in the representation get the point depth.
   *
   * @param point Point to determine depth of.
   * @returns Depth of point.
   */
  PetscInt get_point_depth(const PetscInt point);

  /**
   * Get the depth of the stored representation.
   *
   * @returns Depth of stored graph.
   */
  PetscInt get_depth();

  /**
   * Print the stored representation to stdout.
   */
  void print();

  /**
   * Serialise the stored representation into the provided buffer.
   *
   * @param[in, out] Buffer to serialise the stored representation into.
   */
  void serialise(std::vector<std::byte> &buffer);

  /**
   * Populate this struct instance with the serialised representation provided.
   *
   * @param buffer Serialised representation from which to populate the members
   * of the instance.
   */
  void deserialise(const std::vector<std::byte> &buffer);
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
  virtual void serialise(std::byte *buffer,
                         const std::size_t num_bytes) const override;

  /**
   * Reconstruct the instance from the serialised form.
   *
   * @param buffer Input buffer with serial representation.
   * @param num_bytes Size of input buffer for reference and checking.
   */
  virtual void deserialise(const std::byte *buffer,
                           const std::size_t num_bytes) override;
};

} // namespace NESO::Particles::PetscInterface

#endif
