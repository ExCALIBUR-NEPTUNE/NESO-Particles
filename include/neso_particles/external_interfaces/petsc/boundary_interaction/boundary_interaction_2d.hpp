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
    // An edge has two vertices and each vertex has a coordinate in 2D. Then the
    // normal vector.
    const int ncomp_real = 2 * 2 + 2;

    // label id, global edge point index
    const int ncomp_int = 2;

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

    facet_labels.clear();
    facet_indices.clear();

    MPI_Comm comm_intra = this->sycl_target->comm_pair.comm_intra;
    MPI_Comm comm_inter = this->sycl_target->comm_pair.comm_inter;
    int rank_intra = this->sycl_target->comm_pair.rank_intra;

    std::vector<REAL> node_real;
    std::vector<int> node_int;

    gather_v(local_real, comm_intra, 0, node_real);
    gather_v(local_int, comm_intra, 0, node_int);
    local_real.clear();
    local_int.clear();

    std::vector<REAL> global_real;
    std::vector<int> global_int;

    if (rank_intra == 0) {
      all_gather_v(node_real, comm_inter, global_real);
      all_gather_v(node_int, comm_inter, global_int);
    }
    node_real.clear();
    node_int.clear();

    // On each node the rank where rank_intra == 0 now holds all the boundary
    // edges.

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
