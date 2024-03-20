#ifndef _NESO_PARTICLES_DMPLEX_INTERFACE_HPP_
#define _NESO_PARTICLES_DMPLEX_INTERFACE_HPP_

#include "../../mesh_interface.hpp"
#include "../common/local_claim.hpp"
#include "dmplex_helper.hpp"
#include "petsc_common.hpp"

#include <memory>

namespace NESO::Particles::PetscInterface {

class DMPlexInterface : public HMesh {
protected:
  int ndim;
  int subdivision_order;
  int subdivision_order_offset;
  int cell_count;
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;
  std::vector<int> neighbour_ranks;

  inline void create_mesh_hierarchy() {
    PetscReal global_min[3], global_max[3];
    PETSCCHK(DMGetBoundingBox(this->dmh->dm, global_min, global_max));

    // Compute a set of coarse mesh sizes and dimensions for the mesh hierarchy
    double min_extent = std::numeric_limits<double>::max();
    double max_extent = std::numeric_limits<double>::min();
    REAL global_extents[3];
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const double tmp_global_extent = global_max[dimx] - global_min[dimx];
      global_extents[dimx] = tmp_global_extent;

      min_extent = std::min(min_extent, tmp_global_extent);
      max_extent = std::max(max_extent, tmp_global_extent);
    }
    NESOASSERT(min_extent > 0.0, "Minimum extent is <= 0");
    double coarse_cell_size;
    const double extent_ratio = max_extent / min_extent;
    if (extent_ratio >= 2.0) {
      coarse_cell_size = min_extent;
    } else {
      coarse_cell_size = max_extent;
    }

    std::vector<int> dims(this->ndim);
    std::vector<double> origin(this->ndim);

    int64_t hm_cell_count = 1;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      origin[dimx] = global_min[dimx];
      const int tmp_dim = std::ceil(global_extents[dimx] / coarse_cell_size);
      dims[dimx] = tmp_dim;
      hm_cell_count *= ((int64_t)tmp_dim);
    }

    int64_t global_num_elements;
    int64_t local_num_elements = this->cell_count;
    MPICHK(MPI_Allreduce(&local_num_elements, &global_num_elements, 1,
                         MPI_INT64_T, MPI_SUM, this->comm));

    // compute a subdivision order that would result in the same order of fine
    // cells in the mesh hierarchy as mesh elements in the dmplex
    const double inverse_ndim = 1.0 / ((double)this->ndim);
    const int matching_subdivision_order =
        std::ceil((((double)std::log2(global_num_elements)) -
                   ((double)std::log2(hm_cell_count))) *
                  inverse_ndim);

    // apply the offset to this order and compute the used subdivision order
    this->subdivision_order = std::max(0, matching_subdivision_order +
                                              this->subdivision_order_offset);

    // create the mesh hierarchy
    this->mesh_hierarchy = std::make_shared<MeshHierarchy>(
        this->comm, this->ndim, dims, origin, coarse_cell_size,
        this->subdivision_order);
  }

  inline void
  claim_mesh_hierarchy_cells(ExternalCommon::MHGeomMap &mh_element_map) {

    // local claim cells
    ExternalCommon::LocalClaim local_claim;

    for (int cellx = 0; cellx < this->cell_count; cellx++) {
      const int element_id = this->dmh->get_cell_global_index(cellx);
      const auto bounding_box = this->dmh->get_cell_bounding_box(cellx);
      ExternalCommon::bounding_box_claim(element_id, bounding_box,
                                         this->mesh_hierarchy, local_claim,
                                         mh_element_map);
    }

    // claim cells in the mesh hierarchy
    mesh_hierarchy->claim_initialise();
    for (auto &cellx : local_claim.claim_cells) {
      mesh_hierarchy->claim_cell(cellx,
                                 local_claim.claim_weights[cellx].weight);
    }
    mesh_hierarchy->claim_finalise();
  }

public:
  MPI_Comm comm;
  std::shared_ptr<DMPlexHelper> dmh;

  ~DMPlexInterface(){

  };

  /**
   * TODO
   */
  DMPlexInterface(DM dm, const int subdivision_order_offset = 0,
                  MPI_Comm comm = MPI_COMM_WORLD

                  )
      : comm(comm), subdivision_order_offset(subdivision_order_offset) {

    this->dmh = std::make_shared<DMPlexHelper>(comm, dm);

    PetscInt start, end;
    PETSCCHK(DMPlexGetHeightStratum(this->dmh->dm, 0, &start, &end));
    this->cell_count = static_cast<int>(end) - static_cast<int>(start);
    this->ndim = this->dmh->ndim;

    NESOASSERT((this->ndim > 0) && (this->ndim < 4),
               "Unexpected number of dimensions.");

    this->create_mesh_hierarchy();
    ExternalCommon::MHGeomMap mh_element_map;
    this->claim_mesh_hierarchy_cells(mh_element_map);
  }

  virtual inline void free() override { this->mesh_hierarchy->free(); }

  virtual inline void get_point_in_subdomain(double *point) override {
    NESOASSERT(this->cell_count > 0, "This MPI rank does not own any cells.");
    std::vector<REAL> tmp(this->ndim);
    this->dmh->get_cell_vertex_average(0, tmp);
    for (int dx = 0; dx < this->ndim; dx++) {
      point[dx] = tmp.at(dx);
    }
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

} // namespace NESO::Particles::PetscInterface

#endif
