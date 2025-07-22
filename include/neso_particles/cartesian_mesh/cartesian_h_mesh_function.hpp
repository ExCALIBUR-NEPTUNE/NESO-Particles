#ifndef _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_H_MESH_FUNCTION_HPP_
#define _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_H_MESH_FUNCTION_HPP_

#include "../device_buffers.hpp"
#include "cartesian_h_mesh.hpp"
#include <string>

namespace NESO::Particles {

class CartesianTrajectoryIntersection;

/**
 * Generic function type to represent functions on CartesianHMesh.
 *
 */
class CartesianHMeshFunction {

  friend class CartesianTrajectoryIntersection;

protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  std::shared_ptr<BufferDevice<REAL>> d_dofs;

  /**
   * Create a function on a mesh.
   *
   * @param mesh Host mesh to create function on.
   * @param sycl_target Compute device holding the DOFs.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cell_count Number of locally owned cells on the mesh.
   * @param function_space Type of function to create.
   * @param polynomial_order Polynomial order of function to create.
   * @param element_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  CartesianHMeshFunction(CartesianHMeshSharedPtr mesh,
                         SYCLTargetSharedPtr sycl_target, const int ndim,
                         const int cell_count, const std::string function_space,
                         const int polynomial_order, const int element_group)
      : mesh(mesh), sycl_target(sycl_target), ndim(ndim),
        cell_count(cell_count), element_group(element_group) {
    const int ndof_per_cell = std::pow(polynomial_order + 1, ndim);
    this->d_dofs = std::make_shared<BufferDevice<REAL>>(
        sycl_target, cell_count * ndof_per_cell);
    NESOASSERT(ndim + 1 == mesh->get_ndim(),
               "Only currently implemented for boundary functions.");
    NESOASSERT(function_space == "DG",
               "Only currently implemented for DG0 functions.");
    NESOASSERT(polynomial_order == 0,
               "Only currently implemented for DG0 functions.");
  }

public:
  /// The mesh this function is defined on.
  CartesianHMeshSharedPtr mesh;
  /// Compute device holding the DOFs
  SYCLTargetSharedPtr sycl_target;
  /// The number of dimensions of the space this function is defined in.
  int ndim{0};
  /// The number of cells this function is defined over.
  int cell_count{0};
  /// The type of function, e.g. "DG".
  std::string function_space;
  /// The polynomial order of the function.
  int polynomial_order;
  /// The cells this function is defined over if there is redirection from the
  /// entity index to the cell index.
  std::vector<INT> cells;
  /// If this function corresponds to a boundary group then this entry records
  /// the boundary group.
  int element_group{0};

  CartesianHMeshFunction() = default;
  ~CartesianHMeshFunction() = default;

  /**
   * Create a function on a mesh on the passed entities.
   *
   * @param mesh Host mesh to create function on.
   * @param sycl_target Compute device holding the DOFs.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cells Locally owned mesh entities to create function over.
   * @param function_space Type of function to create.
   * @param polynomial_order Polynomial order of function to create.
   * @param element_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  CartesianHMeshFunction(CartesianHMeshSharedPtr mesh,
                         SYCLTargetSharedPtr sycl_target, const int ndim,
                         const std::vector<INT> &cells,
                         const std::string function_space,
                         const int polynomial_order, const int element_group)
      : CartesianHMeshFunction(mesh, sycl_target, ndim, cells.size(),
                               function_space, polynomial_order,
                               element_group) {
    this->cells = cells;
  }

  /**
   * Write the function to a vtkhdf file.
   *
   * @param filename Output file name which should have vtkhdf extension.
   */
  inline void write_vtkhdf(const std::string filename) {

    NESOASSERT(this->polynomial_order == 0, "Only implemented for DG0.");
    NESOASSERT(this->ndim + 1 == this->mesh->get_ndim(),
               "Only implemented for boundary cells.");

    std::vector<REAL> h_dofs(this->d_dofs->size);
    auto e0 = this->sycl_target->queue.memcpy(
        h_dofs.data(), this->d_dofs->ptr, this->d_dofs->size * sizeof(REAL));

    std::vector<VTK::UnstructuredCell> data;
    data.reserve(this->cells.size());

    e0.wait_and_throw();
    std::size_t index = 0;
    for (auto &cx : this->cells) {
      auto vtkdata = this->mesh->get_vtk_face_cell_data(cx);
      vtkdata.cell_data["u"] = h_dofs.at(index++);
      data.push_back(vtkdata);
    }

    VTK::VTKHDF vtkhdf("mesh2d_face.vtkhdf", mesh->get_comm());
    vtkhdf.write(data, {}, {"u"});
    vtkhdf.close();
  }
};

} // namespace NESO::Particles

#endif
