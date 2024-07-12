#ifndef _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_COMMON_HPP_
#define _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_COMMON_HPP_

#include "../../../containers/blocked_binary_tree.hpp"
#include "../../../loop/particle_loop.hpp"
#include "../../../particle_sub_group.hpp"
#include "../dmplex_interface.hpp"
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace NESO::Particles::PetscInterface {

/**
 * Common implementation elements for identifying intersections between
 * particle trajectories and the mesh boundary. Users probably want to create
 * an instance of classes which inherit from this base, e.g. @ref
 * BoundaryInteraction2D.
 */
class BoundaryInteractionCommon {
protected:
  inline ParticleGroupSharedPtr
  get_particle_group(ParticleGroupSharedPtr iteration_set) {
    return iteration_set;
  }
  inline ParticleGroupSharedPtr
  get_particle_group(ParticleSubGroupSharedPtr iteration_set) {
    return iteration_set->get_particle_group();
  }

  template <typename T>
  inline void check_dat(ParticleGroupSharedPtr particle_group, Sym<T> sym,
                        const int ncomp) {
    if (!particle_group->contains_dat(sym)) {
      ParticleProp prop(sym, ncomp);
      particle_group->add_particle_dat(
          ParticleDat(this->sycl_target, prop, this->mesh->get_cell_count()));
    } else {
      NESOASSERT(
          particle_group->get_dat(sym)->ncomp >= ncomp,
          "Requested dat with sym " + sym.name +
              " exists already with an insufficient number of components");
    }
  }

  std::map<PetscInt, PetscInt> map_label_to_groups;

  std::unique_ptr<MeshHierarchyMapper> mesh_hierarchy_mapper;
  std::unique_ptr<BufferDeviceHost<int>> dh_max_box_size;
  std::unique_ptr<BufferDevice<int>> d_cell_bounds;
  std::unique_ptr<BufferDeviceHost<INT>> dh_mh_cells;
  std::shared_ptr<CellDatConst<int>> cdc_mh_min;
  std::shared_ptr<CellDatConst<int>> cdc_mh_max;
  std::set<INT> required_mh_cells;
  std::set<INT> collected_mh_cells;

  template <typename T>
  inline void prepare_particle_group(std::shared_ptr<T> particle_sub_group) {
    auto particle_group = this->get_particle_group(particle_sub_group);
    NESOASSERT(particle_group->sycl_target == this->sycl_target,
               "Missmatch of sycl targets.");
    const int ndim = this->mesh->get_ndim();
    this->check_dat(particle_group, this->previous_position_sym, ndim);
    this->check_dat(particle_group, this->boundary_position_sym, ndim);
    this->check_dat(particle_group, this->boundary_label_sym, 3);
  }

  inline std::set<PetscInt> get_labels() const {
    std::map<PetscInt, std::set<PetscInt>> bl;
    for (auto &item : this->boundary_groups) {
      for (auto ix : item.second) {
        bl[item.first].insert(ix);
      }
    }

    std::set<PetscInt> labels;
    for (auto &item : bl) {
      for (auto ix : item.second) {
        NESOASSERT(labels.count(ix) == 0, "Label " + std::to_string(ix) +
                                              " exists in the specification of "
                                              "more than one boundary group.");
        labels.insert(ix);
      }
    }

    return labels;
  }

  template <typename T, typename U>
  inline void find_intersections_inner(std::shared_ptr<T> particle_sub_group,
                                       const U &intersect_object) {
    if (intersect_object.boundary_elements_exist()) {
      auto particle_group = this->get_particle_group(particle_sub_group);
      const auto k_ndim = particle_group->position_dat->ncomp;

      auto mesh_hierarchy_device_mapper =
          this->mesh_hierarchy_mapper->get_device_mapper();
      const auto k_cell_bounds = this->d_cell_bounds->ptr;

      particle_loop(
          "BoundaryInteractionCommon::find_intersections_inner",
          particle_sub_group,
          [=](auto B_C, auto B_P, auto PREV_POS, auto CURR_POS) {
            // Get the cells containing the start and end points
            REAL curr_position[3];
            REAL prev_position[3];
            INT cell_starts[3] = {0, 0, 0};
            INT cell_ends[3] = {1, 1, 1};

            for (int dimx = 0; dimx < k_ndim; dimx++) {
              prev_position[dimx] = PREV_POS.at(dimx);
              curr_position[dimx] = CURR_POS.at(dimx);
            }
            mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(
                prev_position, cell_starts);
            mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(
                curr_position, cell_ends);

            // reorder such that start < end for canonical loop ordering
            for (int dimx = 0; dimx < k_ndim; dimx++) {
              const auto a = cell_starts[dimx];
              const auto b = cell_ends[dimx];
              cell_starts[dimx] = KERNEL_MIN(a, b);
              cell_ends[dimx] = KERNEL_MAX(a, b);
            }

            // Truncate the start/end cells to actually be within the range of
            // the grid
            for (int dimx = 0; dimx < k_ndim; dimx++) {
              cell_starts[dimx] = KERNEL_MAX(cell_starts[dimx], 0);
              cell_ends[dimx] =
                  KERNEL_MIN(k_cell_bounds[dimx] - 1, cell_ends[dimx]);
              cell_ends[dimx] += 1;
            }

            // Fill the rest of the iteration set for when k_ndim <3 such that
            // the generic looping works.
            for (int dimx = k_ndim; dimx < 3; dimx++) {
              cell_starts[dimx] = 0;
              cell_ends[dimx] = 1;
            }

            REAL current_distance;
            intersect_object.reset(current_distance);

            // loop over the grid of MH cells
            INT cell_index[3];
            for (cell_index[2] = cell_starts[2]; cell_index[2] < cell_ends[2];
                 cell_index[2]++) {
              for (cell_index[1] = cell_starts[1]; cell_index[1] < cell_ends[1];
                   cell_index[1]++) {
                for (cell_index[0] = cell_starts[0];
                     cell_index[0] < cell_ends[0]; cell_index[0]++) {

                  // convert the cartesian cell index into a mesh heirarchy
                  // index
                  INT mh_tuple[6];
                  mesh_hierarchy_device_mapper.cart_tuple_to_tuple(cell_index,
                                                                   mh_tuple);
                  // convert the mesh hierarchy tuple to linear index
                  const INT linear_index =
                      mesh_hierarchy_device_mapper.tuple_to_linear_global(
                          mh_tuple);
                  intersect_object.find(linear_index, prev_position,
                                        curr_position, current_distance, B_P,
                                        B_C);
                }
              }
            }
          },
          Access::write(this->boundary_label_sym),
          Access::write(this->boundary_position_sym),
          Access::read(this->previous_position_sym),
          Access::read(particle_group->position_dat))
          ->execute();
    }
  }

  /**
   * @param particle_sub_group Collection of particles for which @ref
   * pre_integration has been called.
   */
  template <typename T>
  inline void find_cells(std::shared_ptr<T> particle_sub_group) {

    const int k_INT_MAX = std::numeric_limits<int>::max();
    const int k_INT_MIN = std::numeric_limits<int>::lowest();
    this->cdc_mh_min->fill(k_INT_MAX);
    this->cdc_mh_max->fill(k_INT_MIN);
    auto particle_group = this->get_particle_group(particle_sub_group);
    const auto mesh_hierarchy_device_mapper =
        this->mesh_hierarchy_mapper->get_device_mapper();
    const int k_ndim = this->mesh->get_ndim();
    const int k_cell_count = this->mesh->get_cell_count();

    particle_loop(
        "BoundaryInteractionCommon::find_cells_0", particle_sub_group,
        [=](auto POSITION, auto PREV_POSITION, auto CDC_MAX, auto CDC_MIN) {
          REAL position[3];
          INT cell_cart[3];

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            position[dimx] = POSITION.at(dimx);
          }
          mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(position,
                                                                  cell_cart);
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            CDC_MAX.fetch_max(dimx, 0, cell_cart[dimx]);
            CDC_MIN.fetch_min(dimx, 0, cell_cart[dimx]);
          }
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            position[dimx] = PREV_POSITION.at(dimx);
          }
          mesh_hierarchy_device_mapper.map_to_cart_tuple_no_trunc(position,
                                                                  cell_cart);
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            CDC_MAX.fetch_max(dimx, 0, cell_cart[dimx]);
            CDC_MIN.fetch_min(dimx, 0, cell_cart[dimx]);
          }
        },
        Access::read(particle_group->position_dat),
        Access::read(this->previous_position_sym),
        Access::max(this->cdc_mh_max), Access::min(this->cdc_mh_min))
        ->execute();

    this->dh_max_box_size->h_buffer.ptr[0] = 0;
    this->dh_max_box_size->host_to_device();
    const auto k_cell_bounds = this->d_cell_bounds->ptr;
    int *k_max_ptr = this->dh_max_box_size->d_buffer.ptr;

    int *k_cdc_max = this->cdc_mh_max->device_ptr();
    int *k_cdc_min = this->cdc_mh_min->device_ptr();

    // determine the maximum bounding box size
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(k_cell_count), [=](sycl::id<1> idx) {
                int volume = 1;
                for (int dimx = 0; dimx < k_ndim; dimx++) {

                  const int index = idx * k_ndim + dimx;
                  const int bound_min = KERNEL_MAX(0, k_cdc_min[index]);
                  const int bound_max =
                      KERNEL_MIN(k_cell_bounds[dimx] - 1, k_cdc_max[index]);

                  // Truncated for future computation.
                  k_cdc_min[index] = bound_min;
                  k_cdc_max[index] = bound_max;

                  const int width = bound_max - bound_min + 1;
                  volume *= width;
                }
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    ar(k_max_ptr[0]);
                ar.fetch_max(volume);
              });
        })
        .wait_and_throw();

    this->dh_max_box_size->device_to_host();
    const int k_max_box_size = this->dh_max_box_size->h_buffer.ptr[0];
    this->dh_mh_cells->realloc_no_copy(k_max_box_size * k_cell_count);
    INT *k_mh_cells = this->dh_mh_cells->d_buffer.ptr;

    // populate the mh cells to collect
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(k_cell_count), [=](sycl::id<1> idx) {
                int index = idx;
                INT cell_starts[3] = {0, 0, 0};
                INT cell_ends[3] = {1, 1, 1};

                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  const int index_cell = idx * k_ndim + dimx;
                  const INT bound_min = k_cdc_min[index_cell];
                  const INT bound_max = k_cdc_max[index_cell];
                  cell_starts[dimx] = bound_min;
                  cell_ends[dimx] = bound_max + 1;
                }
                // loop over the cells in the bounding box
                INT cell_index[3];
                for (cell_index[2] = cell_starts[2];
                     cell_index[2] < cell_ends[2]; cell_index[2]++) {
                  for (cell_index[1] = cell_starts[1];
                       cell_index[1] < cell_ends[1]; cell_index[1]++) {
                    for (cell_index[0] = cell_starts[0];
                         cell_index[0] < cell_ends[0]; cell_index[0]++) {

                      // convert the cartesian cell index into a mesh heirarchy
                      // index
                      INT mh_tuple[6];
                      mesh_hierarchy_device_mapper.cart_tuple_to_tuple(
                          cell_index, mh_tuple);
                      // convert the mesh hierarchy tuple to linear index
                      const INT linear_index =
                          mesh_hierarchy_device_mapper.tuple_to_linear_global(
                              mh_tuple);
                      k_mh_cells[index] = linear_index;
                      index += k_cell_count;
                    }
                  }
                }
                // mask off the unused entries
                for (int ix = index; ix < (k_max_box_size * k_cell_count);
                     ix += k_cell_count) {
                  k_mh_cells[ix] = -1;
                }
              });
        })
        .wait_and_throw();

    // Get the required mesh heirarchy cells on the host
    this->dh_mh_cells->device_to_host();
    INT *h_tmp = this->dh_mh_cells->h_buffer.ptr;

    // Get the set of required cells
    std::set<INT> tmp_cells;
    for (int cx = 0; cx < (k_max_box_size * k_cell_count); cx++) {
      const INT cell = h_tmp[cx];
      if (cell > -1) {
        tmp_cells.insert(cell);
      }
    }

    // prune already collected cells
    this->required_mh_cells.clear();
    for (auto cell : tmp_cells) {
      if (this->collected_mh_cells.count(cell) == 0) {
        this->required_mh_cells.insert(cell);
      }
    }
  }

  BoundaryInteractionCommon(
      SYCLTargetSharedPtr sycl_target, DMPlexInterfaceSharedPtr mesh,
      std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
      std::optional<Sym<REAL>> previous_position_sym = std::nullopt,
      std::optional<Sym<REAL>> boundary_position_sym = std::nullopt,
      std::optional<Sym<INT>> boundary_label_sym = std::nullopt)
      : sycl_target(sycl_target), mesh(mesh), boundary_groups(boundary_groups) {

    for (auto &bx : boundary_groups) {
      NESOASSERT(bx.first >= 0, "Group id cannot be negative.");
      for (auto &lx : bx.second) {
        this->map_label_to_groups[lx] = bx.first;
      }
    }

    auto assign_sym = [=](auto &output_sym, auto &input_sym, auto default_sym) {
      if (input_sym != std::nullopt) {
        output_sym = input_sym.value();
      } else {
        output_sym = default_sym;
      }
    };
    assign_sym(this->previous_position_sym, previous_position_sym,
               Sym<REAL>("NESO_PARTICLES_DMPLEX_BOUNDARY_PREV_POS"));
    assign_sym(this->boundary_position_sym, boundary_position_sym,
               Sym<REAL>("NESO_PARTICLES_DMPLEX_BOUNDARY_POS"));
    assign_sym(this->boundary_label_sym, boundary_label_sym,
               Sym<INT>("NESO_PARTICLES_DMPLEX_BOUNDARY_LABEL"));

    const int k_ndim = this->mesh->get_ndim();
    const int k_cell_count = this->mesh->get_cell_count();
    this->cdc_mh_min = std::make_shared<CellDatConst<int>>(
        this->sycl_target, k_cell_count, k_ndim, 1);
    this->cdc_mh_max = std::make_shared<CellDatConst<int>>(
        this->sycl_target, k_cell_count, k_ndim, 1);

    this->mesh_hierarchy_mapper = std::make_unique<MeshHierarchyMapper>(
        this->sycl_target, this->mesh->get_mesh_hierarchy());
    this->dh_max_box_size =
        std::make_unique<BufferDeviceHost<int>>(this->sycl_target, 1);

    const auto mesh_hierarchy_host_mapper =
        this->mesh_hierarchy_mapper->get_host_mapper();

    std::vector<int> h_cell_bounds = {0, 0, 0};
    for (int dimx = 0; dimx < k_ndim; dimx++) {
      const int max_possible_cell = mesh_hierarchy_host_mapper.dims[dimx] *
                                    mesh_hierarchy_host_mapper.ncells_dim_fine;
      h_cell_bounds.at(dimx) = max_possible_cell;
    }

    this->d_cell_bounds =
        std::make_unique<BufferDevice<int>>(this->sycl_target, h_cell_bounds);
    this->dh_mh_cells =
        std::make_unique<BufferDeviceHost<INT>>(this->sycl_target, 1024);
  }


public:

  /// The compute device used to find intersections.
  SYCLTargetSharedPtr sycl_target;
  /// The interface to a DMPlex mesh.
  DMPlexInterfaceSharedPtr mesh;
  /// The map from boundary groups to the labels which form the group.
  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  
  /// The Sym for the particle property which holds the position of each
  /// particle before the positions were updated in a time stepping loop. These
  /// positions are populated on call to @ref pre_integration.
  Sym<REAL> previous_position_sym;
  /// The Sym for the intersection point for the particle trajectory and the
  /// boundary. This property is populated if an intersection is discovered in
  /// a call to @ref post_integration.
  Sym<REAL> boundary_position_sym;
  /// The Sym which holds the metadata information for the intersection point
  /// between the trajectory and the boundary. Component 0 holds a 1 if an
  /// intersection is found. Component 1 holds the group ID identifed for the
  /// intersection. Component 2 holds the global ID of the boundary element for
  /// which the intersection was identified between trajectory and the
  /// boundary.
  Sym<INT> boundary_label_sym;

  /**
   * This method should be called with a collection of particles prior to
   * updating the positions of these particles.
   *
   * @param particles ParticleGroup or ParticleSubGroup of particles whose
   * positions are about to be updated, e.g. in a time stepping operation.
   */
  template <typename T>
  inline void pre_integration(std::shared_ptr<T> particles) {
    prepare_particle_group(particles);
    auto particle_group = this->get_particle_group(particles);
    auto position_dat = particle_group->position_dat;
    const int k_ncomp = position_dat->ncomp;
    const int k_ndim = this->mesh->get_ndim();
    NESOASSERT(
        k_ncomp >= k_ndim,
        "Positions ncomp is smaller than the number of mesh dimensions.");

    particle_loop(
        "BoundaryInteractionCommon::pre_integration", particles,
        [=](auto P, auto PP) {
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            PP.at(dimx) = P.at(dimx);
          }
        },
        Access::read(position_dat->sym),
        Access::write(this->previous_position_sym))
        ->execute();
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
