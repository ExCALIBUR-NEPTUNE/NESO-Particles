#ifndef _NESO_PARTICLES_DMPLEX_INTERFACE_HPP_
#define _NESO_PARTICLES_DMPLEX_INTERFACE_HPP_

#include "../../mesh_hierarchy_data/mesh_hierarchy_data.hpp"
#include "../../mesh_interface.hpp"
#include "../common/local_claim.hpp"
#include "dmplex_helper.hpp"
#include "petsc_common.hpp"

#include <memory>
#include <tuple>

namespace NESO::Particles::PetscInterface {

class DMPlexInterface : public HMesh {
protected:
  bool allocated;
  int ndim;
  int subdivision_order;
  int subdivision_order_offset;
  int cell_count;
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;
  std::vector<int> neighbour_ranks;
  /// Vector of MeshHierarchy cells which are owned by this rank.
  std::vector<INT> owned_mh_cells;
  /// Vector of MeshHierarchy cells which were claimed but are not owned by this
  /// rank.
  std::vector<INT> unowned_mh_cells;

  inline void create_mesh_hierarchy() {
    PetscReal global_min[3], global_max[3];
    PETSCCHK(DMGetBoundingBox(this->dmh->dm, global_min, global_max));

    // Compute a set of coarse mesh sizes and dimensions for the mesh hierarchy
    double min_extent = std::numeric_limits<double>::max();
    double max_extent = std::numeric_limits<double>::lowest();
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

  /**
   *  Find the cells which were claimed by this rank but are acutally owned by
   *  a remote rank
   */
  inline void get_unowned_cells(ExternalCommon::LocalClaim &local_claim) {
    std::stack<INT> owned_cell_stack;
    std::stack<INT> unowned_cell_stack;

    int rank;
    MPICHK(MPI_Comm_rank(this->comm, &rank));

    for (auto &cellx : local_claim.claim_cells) {
      const int owning_rank = this->mesh_hierarchy->get_owner(cellx);
      if (owning_rank == rank) {
        owned_cell_stack.push(cellx);
      } else {
        unowned_cell_stack.push(cellx);
      }
    }
    this->owned_mh_cells.reserve(owned_cell_stack.size());
    while (!owned_cell_stack.empty()) {
      this->owned_mh_cells.push_back(owned_cell_stack.top());
      owned_cell_stack.pop();
    }
    this->unowned_mh_cells.reserve(unowned_cell_stack.size());
    while (!unowned_cell_stack.empty()) {
      this->unowned_mh_cells.push_back(unowned_cell_stack.top());
      unowned_cell_stack.pop();
    }
  }

  inline void
  claim_mesh_hierarchy_cells(ExternalCommon::MHGeomMap &mh_element_map) {

    // local claim cells
    ExternalCommon::LocalClaim local_claim;

    const int num_cells = this->dmh->get_cell_count();
    for (int cellx = 0; cellx < num_cells; cellx++) {
      const auto bounding_box = this->dmh->get_cell_bounding_box(cellx);
      ExternalCommon::bounding_box_claim(cellx, bounding_box,
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
    this->get_unowned_cells(local_claim);
  }

  inline void create_halos(ExternalCommon::MHGeomMap &mh_element_map) {

    std::map<INT, std::vector<DMPlexCellSerialise>> map_cell_dmplex;
    // reserve space
    for (auto item : mh_element_map) {
      const INT mh_cell = item.first;
      map_cell_dmplex[mh_cell].reserve(item.second.size());
    }
    // create the objects
    for (auto item : mh_element_map) {
      const INT mh_cell = item.first;
      for (auto dm_cell : item.second) {
        map_cell_dmplex[mh_cell].push_back(
            this->dmh->get_copyable_cell(dm_cell));
      }
    }
    // send the packed cells to the owning MPI ranks
    MeshHierarchyData::MeshHierarchyContainer mhc(this->mesh_hierarchy,
                                                  map_cell_dmplex);
    // Explicitly gather all cells this rank owns, this should be a lightweight
    // call as the constructor above should have gathered these.
    // TODO add stencil offset
    mhc.gather(this->owned_mh_cells);

    int rank;
    MPICHK(MPI_Comm_rank(this->comm, &rank));

    std::set<PetscInt> dmplex_cell_ids;
    std::list<DMPlexCellSerialise> serialised_halo_cells;

    std::vector<DMPlexCellSerialise> serialised_halo_cells_tmp;
    for (auto mh_cell : this->owned_mh_cells) {
      mhc.get(mh_cell, serialised_halo_cells_tmp);
      for (auto &dmplex_cell : serialised_halo_cells_tmp) {
        if (
            // skip cells this rank owns.
            (dmplex_cell.owning_rank != rank) &&
            // skip cells already collected.
            (!dmplex_cell_ids.count(dmplex_cell.cell_global_id))) {
          serialised_halo_cells.push_back(dmplex_cell);
        }
      }
    }

    // unpack the halo cells into a DMPlex for halo cells
    DM dm_halo;
    auto halo_exists =
        dm_from_serialised_cells(serialised_halo_cells, this->dmh->dm, dm_halo,
                                 this->map_local_lid_remote_lid);
    if (halo_exists) {
      this->dmh_halo = std::make_shared<DMPlexHelper>(PETSC_COMM_SELF, dm_halo);
      std::set<int> remote_ranks;
      int size;
      MPICHK(MPI_Comm_size(this->comm, &size));
      for (auto &ix : this->map_local_lid_remote_lid) {
        const int remote_rank = std::get<0>(ix.second);
        if (remote_rank > -1) {
          NESOASSERT(remote_rank < size, "Bad remote rank.");
          remote_ranks.insert(remote_rank);
        }
      }
      this->neighbour_ranks.insert(this->neighbour_ranks.end(),
                                   remote_ranks.begin(), remote_ranks.end());
    } else {
      this->dmh_halo = nullptr;
    }
    mhc.free();
  }

public:
  MPI_Comm comm;
  std::shared_ptr<DMPlexHelper> dmh;
  std::shared_ptr<DMPlexHelper> dmh_halo;
  std::map<PetscInt, std::tuple<int, PetscInt, PetscInt>>
      map_local_lid_remote_lid;

  ~DMPlexInterface() {
    NESOASSERT(this->allocated == false,
               "DMPlexInterface::free() not called before destruction.");
  };

  /**
   * Create a DMPlex interface object from a DMPlex. Collective on the passed
   * communicator.
   *
   * @param dm DMPlex to create interface from.
   * @param subdivision_order_offset Offset to the subdivision order used to
   * create the mesh hierarchy (default 0).
   * @param comm MPI communicator to use (default MPI_COMM_WORLD).
   */
  DMPlexInterface(DM dm, const int subdivision_order_offset = 0,
                  MPI_Comm comm = MPI_COMM_WORLD)
      : comm(comm), subdivision_order_offset(subdivision_order_offset),
        allocated(true) {

    this->dmh = std::make_shared<DMPlexHelper>(comm, dm);
    this->cell_count = this->dmh->get_cell_count();
    this->ndim = this->dmh->ndim;

    NESOASSERT((this->ndim > 0) && (this->ndim < 4),
               "Unexpected number of dimensions.");

    this->create_mesh_hierarchy();
    ExternalCommon::MHGeomMap mh_element_map;
    this->claim_mesh_hierarchy_cells(mh_element_map);
    this->create_halos(mh_element_map);
  }

  virtual inline void free() override {
    this->mesh_hierarchy->free();
    this->allocated = false;
  }

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

  /**
   * This is a helper function to test that the halos are valid when comparing
   * the topology and coordinates of the halo DMPlex with the original DMPlex
   * the interface was constructed with. Collective on the communicator.
   *
   * @param fatal If true then an error in the validation is fatal.
   * @returns True if no errors are discovered otherwise false.
   */
  inline bool validate_halos(const bool fatal = true) {

    bool valid = true;
    auto lambda_assert_true = [&](const bool cond) {
      valid = valid && cond;
      if (fatal) {
        NESOASSERT(cond, "Conditional failed in validate_halos.");
      }
    };
    auto lambda_assert_eq = [&](const auto a, const auto b) {
      valid = valid && (a == b);
      if (fatal) {
        NESOASSERT(a == b, "Conditional failed in validate_halos.");
      }
    };

    auto dm = this->dmh->dm;
    int rank, size;
    MPICHK(MPI_Comm_rank(this->comm, &rank));
    MPICHK(MPI_Comm_rank(this->comm, &size));

    PetscInt point_start, point_end;
    PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));

    const int num_local_points = point_end - point_start;
    int max_global_point = 0;
    for (int ix = 0; ix < num_local_points; ix++) {
      max_global_point =
          std::max(max_global_point,
                   this->dmh->get_point_global_index(ix + point_start));
    }
    int num_points;
    MPICHK(MPI_Allreduce(&max_global_point, &num_points, 1, MPI_INT, MPI_MAX,
                         MPI_COMM_WORLD));
    num_points++;

    const int num_components = 1 + 1 + 4 + 1;
    auto I = [=](const int rx, const int cx) {
      return rx * num_components + cx;
    };
    const int num_components_real = 8;
    auto F = [=](const int rx, const int cx) {
      return rx * num_components_real + cx;
    };

    std::vector<int> int_data(num_points * num_components);
    std::vector<int> int_rdata(num_points * num_components);
    std::fill(int_data.begin(), int_data.end(),
              std::numeric_limits<int>::lowest());
    std::fill(int_rdata.begin(), int_rdata.end(),
              std::numeric_limits<int>::lowest());
    std::vector<double> real_data(num_points * num_components_real);
    std::vector<double> real_rdata(num_points * num_components_real);
    std::fill(real_data.begin(), real_data.end(),
              std::numeric_limits<double>::lowest());
    std::fill(real_rdata.begin(), real_rdata.end(),
              std::numeric_limits<double>::lowest());

    for (int ix = 0; ix < num_local_points; ix++) {
      PetscInt point = ix + point_start;
      const int global_point = this->dmh->get_point_global_index(point, true);
      if (global_point > -1) {
        // INT data
        // Get the point depth
        PetscInt depth;
        PETSCCHK(DMPlexGetPointDepth(dm, point, &depth));

        int_data.at(I(global_point, 0)) = depth;

        PetscInt cone_size;
        PETSCCHK(DMPlexGetConeSize(dm, point, &cone_size));
        int_data.at(I(global_point, 1)) = cone_size;

        lambda_assert_true(cone_size < 5);

        const PetscInt *cone;
        PETSCCHK(DMPlexGetCone(dm, point, &cone));

        for (int cx = 0; cx < cone_size; cx++) {
          int_data.at(I(global_point, 2 + cx)) =
              this->dmh->get_point_global_index(cone[cx]);
        }

        int_data.at(I(global_point, 6)) = rank;

        // REAL data
        PetscBool is_dg;
        PetscInt num_coords;
        const PetscScalar *array;
        PetscScalar *coords = nullptr;
        PETSCCHK(DMPlexGetCellCoordinates(dm, point, &is_dg, &num_coords,
                                          &array, &coords));
        lambda_assert_true(num_coords <= 8);
        for (int cx = 0; cx < num_coords; cx++) {
          real_data.at(F(global_point, cx)) = coords[cx];
        }
        PETSCCHK(DMPlexRestoreCellCoordinates(dm, point, &is_dg, &num_coords,
                                              &array, &coords));
      }
    }

    MPICHK(MPI_Allreduce(int_data.data(), int_rdata.data(), int_rdata.size(),
                         MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    MPICHK(MPI_Allreduce(real_data.data(), real_rdata.data(), real_rdata.size(),
                         MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    if (this->dmh_halo) {
      // create map from halo points to original global points
      std::map<PetscInt, PetscInt> map_halo_to_global;
      std::map<PetscInt, int> map_halo_to_rank;
      for (auto ix : this->map_local_lid_remote_lid) {
        map_halo_to_global[ix.first] = std::get<2>(ix.second);
        map_halo_to_rank[ix.first] = std::get<0>(ix.second);
      }

      auto dm_halo = this->dmh_halo->dm;
      PetscInt hpoint_start, hpoint_end;
      PETSCCHK(DMPlexGetChart(dm_halo, &hpoint_start, &hpoint_end));

      for (int hpoint = hpoint_start; hpoint < hpoint_end; hpoint++) {
        // Original global point index
        const PetscInt global_point = map_halo_to_global.at(hpoint);

        PetscInt depth;
        PETSCCHK(DMPlexGetPointDepth(dm_halo, hpoint, &depth));

        PetscInt cone_size;
        PETSCCHK(DMPlexGetConeSize(dm_halo, hpoint, &cone_size));
        const PetscInt *cone;
        PETSCCHK(DMPlexGetCone(dm_halo, hpoint, &cone));

        lambda_assert_eq(int_rdata.at(I(global_point, 0)), depth);
        lambda_assert_eq(int_rdata.at(I(global_point, 1)), cone_size);
        for (int cx = 0; cx < cone_size; cx++) {
          lambda_assert_eq(int_rdata.at(I(global_point, 2 + cx)),
                           map_halo_to_global.at(cone[cx]));
        }

        const int rank_held = map_halo_to_rank.at(hpoint);
        if (depth == 2) {
          lambda_assert_eq(rank_held, int_rdata.at(I(global_point, 6)));
        }

        // REAL data
        PetscBool is_dg;
        PetscInt num_coords;
        const PetscScalar *array;
        PetscScalar *coords = nullptr;
        PETSCCHK(DMPlexGetCellCoordinates(dm_halo, hpoint, &is_dg, &num_coords,
                                          &array, &coords));
        lambda_assert_true(num_coords <= 8);

        std::set<std::tuple<double, double>> correct, to_test;
        for (int cx = 0; cx < num_coords; cx += 2) {
          correct.insert({real_rdata.at(F(global_point, cx)),
                          real_rdata.at(F(global_point, cx + 1))});
          to_test.insert({coords[cx], coords[cx + 1]});
        }
        lambda_assert_eq(to_test, correct);

        PETSCCHK(DMPlexRestoreCellCoordinates(dm_halo, hpoint, &is_dg,
                                              &num_coords, &array, &coords));
      }
    }

    return valid;
  }
};

typedef std::shared_ptr<DMPlexInterface> DMPlexInterfaceSharedPtr;

} // namespace NESO::Particles::PetscInterface

#endif
