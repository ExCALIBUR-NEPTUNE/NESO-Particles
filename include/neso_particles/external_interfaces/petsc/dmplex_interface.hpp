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

  void create_mesh_hierarchy();

  /**
   *  Find the cells which were claimed by this rank but are acutally owned by
   *  a remote rank
   */
  void get_unowned_cells(ExternalCommon::LocalClaim &local_claim);

  void claim_mesh_hierarchy_cells(ExternalCommon::MHGeomMap &mh_element_map);

  void create_halos(ExternalCommon::MHGeomMap &mh_element_map);

public:
  MPI_Comm comm;
  std::shared_ptr<DMPlexHelper> dmh;
  std::shared_ptr<DMPlexHelper> dmh_halo;
  std::map<PetscInt, std::tuple<int, PetscInt, PetscInt>>
      map_local_lid_remote_lid;

  virtual ~DMPlexInterface() {
    if (this->allocated == true) {
      nprint("DMPlexInterface::free() not called before destruction.");
    }
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
                  MPI_Comm comm = MPI_COMM_WORLD);

  virtual void free() override;
  virtual void get_point_in_subdomain(double *point) override;
  virtual std::vector<int> &get_local_communication_neighbours() override;
  virtual MPI_Comm get_comm() override;
  virtual int get_ndim() override;
  virtual std::vector<int> &get_dims() override;
  virtual int get_subdivision_order() override;
  virtual int get_cell_count() override;
  virtual double get_cell_width_coarse() override;
  virtual double get_cell_width_fine() override;
  virtual double get_inverse_cell_width_coarse() override;
  virtual double get_inverse_cell_width_fine() override;
  virtual int get_ncells_coarse() override;
  virtual int get_ncells_fine() override;
  virtual std::shared_ptr<MeshHierarchy> get_mesh_hierarchy() override;

  /**
   * This is a helper function to test that the halos are valid when comparing
   * the topology and coordinates of the halo DMPlex with the original DMPlex
   * the interface was constructed with. Collective on the communicator.
   *
   * @param fatal If true then an error in the validation is fatal.
   * @returns True if no errors are discovered otherwise false.
   */
  bool validate_halos(const bool fatal = true);
};

typedef std::shared_ptr<DMPlexInterface> DMPlexInterfaceSharedPtr;

} // namespace NESO::Particles::PetscInterface

#endif
