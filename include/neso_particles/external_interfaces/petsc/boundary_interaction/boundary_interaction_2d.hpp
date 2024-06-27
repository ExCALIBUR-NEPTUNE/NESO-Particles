#ifndef _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_2D_HPP_
#define _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_2D_HPP_

#include "boundary_interaction_common.hpp"
#include <cmath>
#include <numeric>

namespace NESO::Particles::PetscInterface {

struct BoundaryInteractionHit {
  REAL *d_real;
  int face_set;
};

/**
 * TODO
 */
class BoundaryInteraction2D : public BoundaryInteractionCommon {
protected:
  std::unique_ptr<BufferDevice<REAL>> d_data_real;
  std::unique_ptr<BlockedBinaryTree<int, BoundaryInteractionHit, 8>>
      d_map_gid_data;

public:
  /**
   * TODO
   * collective
   */
  template <typename... T>
  BoundaryInteraction2D(T... args) : BoundaryInteractionCommon(args...) {

    // Get the boundary labels this instance should detect interactions with.
    auto labels = this->get_labels();

    // map from label to petsc point indices in the dm for the facets
    auto face_sets = this->mesh->dmh->get_face_sets();

    // Keep and flatten the points/labels of interest
    std::vector<PetscInt> facet_labels;
    std::vector<PetscInt> facet_indices;
    for (auto &item : face_sets) {
      if (labels.count(item.first)) {
        facet_labels.reserve(facet_labels.size() + item.second.size());
        facet_indices.reserve(facet_indices.size() + item.second.size());
        for (auto &fx : item.second) {
          // push back the label
          facet_labels.push_back(item.first);
          // push back the petsc point index
          facet_indices.push_back(fx);
        }
      }
    }
    face_sets.clear();

    int num_facets_local = facet_labels.size();

    MPI_Comm comm = this->sycl_target->comm_pair.comm_parent;
    int size = this->sycl_target->comm_pair.size_parent;
    int rank = this->sycl_target->comm_pair.rank_parent;

    std::vector<int> recv_counts(size);
    std::fill(recv_counts.begin(), recv_counts.end(), 1);
    std::vector<int> displacements(size);
    std::iota(displacements.begin(), displacements.end(), 0);
    std::vector<int> facet_counts(size);

    // Get on each rank the number of facets all the other ranks have
    MPICHK(MPI_Allgatherv(&num_facets_local, 1, MPI_INT, facet_counts.data(),
                          recv_counts.data(), displacements.data(), MPI_INT,
                          comm));

    NESOASSERT(facet_counts.at(rank) == num_facets_local,
               "MPI_Allgatherv did not collect this rank's contribution");

    const int num_facets_global =
        std::reduce(facet_counts.begin(), facet_counts.end());

    std::exclusive_scan(facet_counts.begin(), facet_counts.end(),
                        displacements.begin(), 0);

    // An edge has two vertices and each vertex has a coordinate in 2D. Then the
    // normal vector.
    const int ncomp_real = 2 * 2 + 2;

    // label id, global edge point index
    const int ncomp_int = 2;

    // Compute the displacements for the real and int data
    std::vector<int> displacements_real(size);
    std::vector<int> displacements_int(size);
    std::vector<int> recv_counts_real(size);
    std::vector<int> recv_counts_int(size);
    for (int ix = 0; ix < size; ix++) {
      displacements_real.at(ix) = ncomp_real * displacements.at(ix);
      displacements_int.at(ix) = ncomp_int * displacements.at(ix);
      recv_counts_real.at(ix) = ncomp_real * recv_counts.at(ix);
      recv_counts_int.at(ix) = ncomp_int * recv_counts.at(ix);
    }

    // space to store the local contributions
    std::vector<REAL> local_real(num_facets_local * ncomp_real);
    std::vector<int> local_int(num_facets_local * ncomp_int);

    // collect the local edges to send
    std::vector<std::vector<REAL>> coords;
    for (int ix = 0; ix < num_facets_local; ix++) {
      const PetscInt index = facet_indices.at(ix);
      // Collect the vertex coords
      this->mesh->dmh->get_generic_vertices(index, coords);
      NESOASSERT(coords.size() == 2,
                 "Expected an edge to only have two vertices.");
      NESOASSERT(coords.at(0).size() == 2,
                 "Expected edge vertex to be embedded in 2D.");
      NESOASSERT(coords.at(1).size() == 2,
                 "Expected edge vertex to be embedded in 2D.");

      const REAL x0 = coords.at(0).at(0);
      const REAL y0 = coords.at(0).at(1);
      const REAL x1 = coords.at(1).at(0);
      const REAL y1 = coords.at(1).at(1);

      // compute the normal to the facet
      const REAL dx = x1 - x0;
      const REAL dy = y1 - y0;
      const REAL n0t = -dy;
      const REAL n1t = dx;
      const REAL l = 1.0 / std::sqrt(n0t * n0t + n1t * n1t);
      const REAL n0 = n0t * l;
      const REAL n1 = n1t * l;

      local_real.at(ix * ncomp_real + 0) = x0;
      local_real.at(ix * ncomp_real + 1) = y0;
      local_real.at(ix * ncomp_real + 2) = x1;
      local_real.at(ix * ncomp_real + 3) = y1;
      local_real.at(ix * ncomp_real + 4) = n0;
      local_real.at(ix * ncomp_real + 5) = n1;

      // collect the label index and edge global id
      const PetscInt facet_global_id =
          this->mesh->dmh->get_point_global_index(index);
      local_int.at(ix * ncomp_int + 0) = facet_labels.at(ix);
      local_int.at(ix * ncomp_int + 1) = facet_global_id;
    }

    // space to store all the edges from all ranks
    std::vector<REAL> global_real(num_facets_global * ncomp_real);
    std::vector<int> global_int(num_facets_global * ncomp_int);

    // Gather all the edge data on all the ranks
    MPICHK(MPI_Allgatherv(local_real.data(), num_facets_local * ncomp_real,
                          map_ctype_mpi_type<REAL>(), global_real.data(),
                          recv_counts_real.data(), displacements_real.data(),
                          map_ctype_mpi_type<REAL>(), comm));
    MPICHK(MPI_Allgatherv(local_int.data(), num_facets_local * ncomp_int,
                          MPI_INT, global_int.data(), recv_counts_int.data(),
                          displacements_int.data(), MPI_INT, comm));

    // Create the data structures for kernels.
    // Push the actual data onto the device.
    this->d_data_real =
        std::make_unique<BufferDevice<REAL>>(this->sycl_target, global_real);

    // Alloc the map from global ids to data
    this->d_map_gid_data =
        std::make_unique<BlockedBinaryTree<int, BoundaryInteractionHit, 8>>(
            this->sycl_target);
    // Build the entries
    for (int ix = 0; ix < num_facets_global; ix++) {
      const int face_set = global_int.at(ncomp_int * ix + 0);
      const int global_id = global_int.at(ncomp_int * ix + 1);
      BoundaryInteractionHit v;
      v.d_real = this->d_data_real->ptr + ncomp_real * ix;
      v.face_set = face_set;
      this->d_map_gid_data->add(global_id, v);
    }

    // TODO Create an overlay mesh
    // TODO create map from overlay mesh cells to global ids of edges
    // Maybe redo the LUT to be directly to the data in an overlay cell to have
    // 1 redirection instead of \approx 3
    // TODO do this map assembly per overlay cell in a separate function so can
    // add more data to the device only as needed otherwise we will end up
    // pushing all the edges for the whole domain onto the device from the
    // start which will be expensive?
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
