#ifndef _NESO_PARTICLES_DMPLEX_INTERFACE_HPP_
#define _NESO_PARTICLES_DMPLEX_INTERFACE_HPP_

#include "../../mesh_interface.hpp"
#include "petsc_common.hpp"
#include <petscdmplex.h>

namespace NESO::Particles::PETScInterface {

class DMPlexInterface : public HMesh {
protected:
  int ndim;
  int subdivision_order;
  int subdivision_order_offset;
  int cell_count;
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;
  std::vector<int> neighbour_ranks;

public:
  MPI_Comm comm;
  DM dm;

  ~DMPlexInterface(){

  };

  /**
   * TODO
   */
  DMPlexInterface(DM dm, const int subdivision_order_offset = 0,
                  MPI_Comm comm = MPI_COMM_WORLD

                  )
      : comm(comm), dm(dm), subdivision_order_offset(subdivision_order_offset) {

    // TODO set subdivision_order

    PetscInt start, end;
    PETSCCHK(DMPlexGetHeightStratum(this->dm, 0, &start, &end));
    this->cell_count = static_cast<int>(end);
    PetscInt dim;
    PETSCCHK(DMGetCoordinateDim(this->dm, &dim));
    this->ndim = static_cast<int>(dim);







    int64_t global_num_elements;
    int64_t local_num_elements = this->cell_count;
    MPICHK(MPI_Allreduce(&local_num_elements, &global_num_elements, 1,
                         MPI_INT64_T, MPI_SUM, this->comm));

    // compute a subdivision order that would result in the same order of fine
    // cells in the mesh hierarchy as mesh elements in Nektar++
    const double inverse_ndim = 1.0 / ((double)this->ndim);
    const int matching_subdivision_order =
        std::ceil((((double)std::log(global_num_elements)) -
                   ((double)std::log(hm_cell_count))) *
                  inverse_ndim);

    // apply the offset to this order and compute the used subdivision order
    this->subdivision_order = std::max(0, matching_subdivision_order +
                                              this->subdivision_order_offset);

  }

  virtual inline void free() override {
    // TODO
  }

  virtual inline void get_point_in_subdomain(double *point) override {
    // TODO
  }

  virtual inline std::vector<int> &
  get_local_communication_neighbours() override {
    return this->neighbour_ranks;
  }
  virtual inline MPI_Comm get_comm() override { return this->comm; };
  virtual inline int get_ndim() override { return this->ndim; };
  virtual inline std::vector<int> &get_dims() override {
    return this->mesh_hierarchy->dims;
  };
  virtual inline int get_subdivision_order() override {
    return this->subdivision_order;
  };
  virtual inline int get_cell_count() override { return this->cell_count; };
  virtual inline double get_cell_width_coarse() override {
    return this->mesh_hierarchy->cell_width_coarse;
  };
  virtual inline double get_cell_width_fine() override {
    return this->mesh_hierarchy->cell_width_fine;
  };
  virtual inline double get_inverse_cell_width_coarse() override {
    return this->mesh_hierarchy->inverse_cell_width_coarse;
  };
  virtual inline double get_inverse_cell_width_fine() override {
    return this->mesh_hierarchy->inverse_cell_width_fine;
  };
  virtual inline int get_ncells_coarse() override {
    return this->mesh_hierarchy->ncells_coarse;
  };
  virtual inline int get_ncells_fine() override {
    return this->mesh_hierarchy->ncells_fine;
  };
  virtual inline std::shared_ptr<MeshHierarchy> get_mesh_hierarchy() override {
    return this->mesh_hierarchy;
  }
};

} // namespace NESO::Particles::PETScInterface

#endif
