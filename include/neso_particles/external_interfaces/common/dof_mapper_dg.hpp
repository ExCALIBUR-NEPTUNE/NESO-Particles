#ifndef _NESO_PARTICLES_DOF_MAPPER_DG_H_
#define _NESO_PARTICLES_DOF_MAPPER_DG_H_
#include "../../compute_target.hpp"
#include "../../containers/cell_dat_const.hpp"
#include "../../typedefs.hpp"
#include <vector>

namespace NESO::Particles::ExternalCommon {

/**
 * Device copyable type for mapping indices in the internal ordering to the
 * external ordering and vice versa.
 */
struct DOFMapperDGDeviceMapper {
  int num_cells_local;
  int num_dofs_per_cell;
  int *d_map_to_index;

  /**
   * @param cell Local index of cell to get DOF ordering for.
   * @param dof Cell local index of DOF to get the external index of.
   * @returns External index for cell and DOF.
   */
  inline int get(const int cell, const int dof) const {
    return this->d_map_to_index[cell * num_dofs_per_cell + dof];
  }

  /**
   * Copy DOF value from a buffer in the internal ordering to a buffer in the
   * external ordering.
   *
   * @param[in] cell Cell containing DOF to copy.
   * @param[in] dof Cell local DOF index for value to copy.
   * @param[in] source Pointer to buffer containing all DOFs.
   * @param[in, out] destination Pointer to destination buffer for all DOFS.
   */
  inline void copy_to_external(const int cell, const int dof,
                               const REAL *RESTRICT source,
                               REAL *RESTRICT destination) const {
    const int index_source = cell * num_dofs_per_cell + dof;
    const int index_dst = this->get(cell, dof);
    destination[index_dst] = source[index_source];
  }

  /**
   * Copy DOF value from a buffer containing DOFs in the external ordering to a
   * buffer containing DOFs in the internal ordering.
   *
   * @param[in] cell Index of cell to copy DOF value for.
   * @param[in] dof Cell local DOF index to copy.
   * @param[in] source Pointer to buffer containing all DOFs.
   * @param[in, out] destination Pointer to destination buffer for all DOFs.
   */
  inline void copy_from_external(const int cell, const int dof,
                                 const REAL *RESTRICT source,
                                 REAL *RESTRICT destination) const {
    const int index_dst = cell * num_dofs_per_cell + dof;
    const int index_source = this->get(cell, dof);
    destination[index_dst] = source[index_source];
  }
};

/**
 * Holds DOF values in an internal representation on the device. DOF ordering on
 * the device may be different to the host.
 */
class DOFMapperDG {
protected:
  std::vector<int> map_to_index;
  std::unique_ptr<BufferDevice<int>> d_map_to_index;
  std::unique_ptr<BufferDevice<REAL>> d_dofs;
  bool device_valid;

public:
  SYCLTargetSharedPtr sycl_target;
  int num_cells_local;
  int num_dofs_per_cell;

  DOFMapperDG() = default;

  /**
   * Create a DOF mapper on a compute device for a given number of cells and a
   * given number of DOFs per cell.
   *
   * @param sycl_target Compute device to store DOFs on.
   * @param num_cells_local Number of cells to create a store for.
   * @param num_dofs_per_cell Number of DOFs per cell.
   */
  DOFMapperDG(SYCLTargetSharedPtr sycl_target, const int num_cells_local,
              const int num_dofs_per_cell);

  /**
   * Retrieve the internal index for a given cell and DOF index.
   *
   * @param cell Index of the cell to index.
   * @param dof Index of the DOF to index.
   * @returns Linearised internal index in the store for the cell and DOF.
   */
  int index(const int cell, const int dof) const;

  /**
   * Set the external linearised DOF index (global) for a given cell and DOF
   * index (within the cell).
   *
   * @param cell The cell index to set the linearised global index for.
   * @param dof The cell local index of the DOF to set the index for.
   * @param index The global linearised index of the DOF.
   */
  void set(const int cell, const int dof, const int index);

  /**
   * Retrieve the external index for a given cell and DOF index.
   *
   * @param cell Index of the cell to index.
   * @param dof Index of the DOF to index.
   * @returns Linearised external index in the store for the cell and DOF.
   */
  int get(const int cell, const int dof) const;

  /**
   * @returns A device copyable helper type for mapping between internal and
   * external DOF orderings.
   */
  DOFMapperDGDeviceMapper get_device_mapper();

  /**
   * Copy DOF values from the DOFs stored in a CellDatConst, stored in the
   * internal ordering, to an external buffer with DOFs in the external
   * ordering.
   *
   * @param[in] cell_dat_const Source CellDatConst containing DOFs in the
   * internal ordering.
   * @param[in, out] h_external_dofs Output location to copy DOFs into.
   * @param[in] es EventStack to push the copy operation onto.
   */
  void copy_to_external(CellDatConstSharedPtr<REAL> cell_dat_const,
                        REAL *h_external_dofs, EventStack &es);

  /**
   *
   * @param[in, out] cell_dat_const  Destination CellDatConst containing DOFs in
   * the internal ordering.
   * @param[in] h_external_dofs Source buffer containing DOFs in the external
   * ordering.
   * @param[in] es EventStack to push the copy operation onto.
   */
  void copy_from_external(CellDatConstSharedPtr<REAL> cell_dat_const,
                          REAL *h_external_dofs, EventStack &es);
};

} // namespace NESO::Particles::ExternalCommon

#endif
