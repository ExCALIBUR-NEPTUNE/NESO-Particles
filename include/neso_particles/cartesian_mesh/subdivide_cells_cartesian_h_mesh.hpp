#ifndef _NESO_PARTICLES_CARTESIAN_MESH_SUBDIVIDE_CELLS_CARTESIAN_HPP_
#define _NESO_PARTICLES_CARTESIAN_MESH_SUBDIVIDE_CELLS_CARTESIAN_HPP_

#include "../compute_target.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"
#include "cartesian_h_mesh.hpp"

namespace NESO::Particles {

/**
 * Implementation to subdivide each cell of a CartesianHMesh n times per
 * dimension.
 */
class SubdivideCellsCartesianHMesh {
protected:
  SYCLTargetSharedPtr sycl_target;
  CartesianHMeshSharedPtr mesh;
  std::vector<int, HostAllocator<int>> h_num_subdivisions;
  std::vector<int> h_num_collision_cells;
  EventStack event_stack;

  std::shared_ptr<BufferDevice<int>> d_num_subdivisions;
  std::shared_ptr<BufferDevice<int>> d_num_collision_cells;
  std::shared_ptr<BufferDevice<REAL>> d_origins;
  std::shared_ptr<BufferDevice<REAL>> d_inverse_cell_widths;

  template <int ndim, typename GROUP_TYPE>
  static inline void
  map_inner(std::shared_ptr<GROUP_TYPE> particle_sub_group, Sym<INT> sym_name,
            const int sym_component, const int k_num_cells,
            int const *const RESTRICT k_num_subdivisions,
            REAL const *const RESTRICT k_origins,
            REAL const *const RESTRICT k_inverse_cell_widths) {

    auto particle_group = get_particle_group(particle_sub_group);
    particle_loop(
        "SubdivideCellsCartesianHMesh::map", particle_sub_group,
        [=](auto INDEX, auto POS, auto VCELL) {
          REAL p[ndim];
          for (int dx = 0; dx < ndim; dx++) {
            p[dx] = POS.at(dx);
          }
          const auto cell = INDEX.cell;
          const int num_cells_per_dim = 1 << k_num_subdivisions[cell];

          int scells[ndim];
          for (int dx = 0; dx < ndim; dx++) {
            const REAL p_shifted = p[dx] - k_origins[dx * k_num_cells + cell];
            const REAL p_shifted_real_bin =
                p_shifted * k_inverse_cell_widths[cell];
            const int scell_int = Kernel::clamp(
                static_cast<int>(p_shifted_real_bin), 0, num_cells_per_dim - 1);
            scells[dx] = scell_int;
          }

          int vcell = scells[ndim - 1];
          for (int dx = ndim - 2; dx >= 0; dx--) {
            vcell *= num_cells_per_dim;
            vcell += scells[dx];
          }

          VCELL.at(sym_component) = vcell;
        },
        Access::read(ParticleLoopIndex{}),
        Access::read(particle_group->position_dat), Access::write(sym_name))
        ->execute();
  }

public:
  SubdivideCellsCartesianHMesh() = delete;

  /// Disable (implicit) copies.
  SubdivideCellsCartesianHMesh(const SubdivideCellsCartesianHMesh &st) = delete;
  /// Disable (implicit) copies.
  SubdivideCellsCartesianHMesh &
  operator=(SubdivideCellsCartesianHMesh const &a) = delete;

  ~SubdivideCellsCartesianHMesh();

  /**
   * Create instance on compute device and mesh.
   *
   * @param sycl_target Compute device.
   * @param mesh CartesianHMesh instance.
   */
  SubdivideCellsCartesianHMesh(SYCLTargetSharedPtr sycl_target,
                               CartesianHMeshSharedPtr mesh);

  /**
   * Set the number of subdivisions in each cell per dimension.
   *
   * @param num_subdivisions Number of subdivisions.
   */
  void set_num_subdivisions(const std::vector<int> &num_subdivisions);

  /**
   * Map particles to subdivision cells.
   *
   * @param particle_sub_group Particle{Sub}Group of particles.
   * @param sym_name Output Sym in which to store subdivision cell.
   * @param sym_component Output component in which to store subdivision cell.
   */
  template <typename GROUP_TYPE>
  void map(std::shared_ptr<GROUP_TYPE> particle_sub_group, Sym<INT> sym_name,
           const int sym_component) {
    this->event_stack.wait();

    auto particle_group = get_particle_group(particle_sub_group);

    NESOASSERT(particle_group->sycl_target.get() == this->sycl_target.get(),
               "SYCLTarget missmatch,");
    NESOASSERT(particle_group->domain->mesh.get() == this->mesh.get(),
               "Miss-match of particles mesh and the mesh this instance was "
               "constructed with.");
    NESOASSERT(particle_group->contains_dat(sym_name),
               "Output sym not in ParticleGroup");
    NESOASSERT(sym_component > -1, "Bad output sym component.");
    NESOASSERT(particle_group->get_dat(sym_name)->ncomp > sym_component,
               "Output sym does not have enough components.");

    const int ndim = this->mesh->get_ndim();
    const auto num_cells = mesh->get_cell_count();

    if (ndim == 1) {
      map_inner<1>(particle_sub_group, sym_name, sym_component, num_cells,
                   this->d_num_subdivisions->ptr, this->d_origins->ptr,
                   this->d_inverse_cell_widths->ptr);
    } else if (ndim == 2) {
      map_inner<2>(particle_sub_group, sym_name, sym_component, num_cells,
                   this->d_num_subdivisions->ptr, this->d_origins->ptr,
                   this->d_inverse_cell_widths->ptr);
    } else {
      map_inner<3>(particle_sub_group, sym_name, sym_component, num_cells,
                   this->d_num_subdivisions->ptr, this->d_origins->ptr,
                   this->d_inverse_cell_widths->ptr);
    }
  }

  /**
   * This is a helper method that computes (2^s)^d for each cell where s is the
   * number of subdivisions and d is the number of dimensions.
   *
   * @returns Total number of subdivided cells.
   */
  const std::vector<int> &get_num_subdivision_cells();
};

extern template void SubdivideCellsCartesianHMesh::map(
    std::shared_ptr<ParticleGroup> particle_sub_group, Sym<INT> sym_name,
    const int sym_component);

extern template void SubdivideCellsCartesianHMesh::map(
    std::shared_ptr<ParticleSubGroup> particle_sub_group, Sym<INT> sym_name,
    const int sym_component);
} // namespace NESO::Particles

#endif
