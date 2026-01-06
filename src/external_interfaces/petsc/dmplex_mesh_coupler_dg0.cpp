#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/external_interfaces/petsc/dmplex_mesh_coupler_dg0.hpp>

namespace NESO::Particles::PetscInterface {

DMPlexMeshCouplerDG0::DMPlexMeshCouplerDG0(
    MPI_Comm comm, DM B_dmplex,
    std::vector<std::vector<DMPlexMeshCouplerDG0MapEntry>> &coupling_map)
    :
      comm(comm), B_dmplex(B_dmplex) {

  const std::size_t num_local_cells_A = coupling_map.size();

  // Need to find the ranks which own the cells in the B mesh for each cell we
  // were passed in the A mesh.

  MPI_Comm B_comm = MPI_COMM_NULL;
  PETSCCHK(PetscObjectGetComm((PetscObject) B_dmplex, &B_comm));
  DMPlexHelper dmh_B(B_comm, B_dmplex);




  
  // For each remote rank create an edge in the neighbour graph.





  // Reverse the neighbour graph direction for the backwards transfer.



  
  // Exchange the cell counts for each edge (can use the forward graph)




  // Exchange the cell indices for each edge


}

void DMPlexMeshCouplerDG0::forward_transfer(std::vector<REAL> &dofs_A,
                                            std::vector<REAL> &dofs_B) {}

void DMPlexMeshCouplerDG0::backward_transfer(std::vector<REAL> &dofs_B,
                                             std::vector<REAL> &dofs_A) {}

void DMPlexMeshCouplerDG0::free() {
  if (this->comm_forward != MPI_COMM_NULL) {
    MPICHK(MPI_Comm_free(&this->comm_forward));
    this->comm_forward = MPI_COMM_NULL;
  }
  if (this->comm_backward != MPI_COMM_NULL) {
    MPICHK(MPI_Comm_free(&this->comm_backward));
    this->comm_backward = MPI_COMM_NULL;
  }
}

} // namespace NESO::Particles::PetscInterface

#endif
