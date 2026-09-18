#ifndef _NESO_PARTICLES_PETSC_INTERFACE_HPP_
#define _NESO_PARTICLES_PETSC_INTERFACE_HPP_

// clang-format off

/**
 * @defgroup external_interfaces_petsc PETSc
 * @details Implementation that interfaces with PETSc and DMPlex.
 */

/**
 * @defgroup external_interfaces_petsc_dmplex_helper_functions DMPlex Helper
 * Functions
 * @ingroup external_interfaces_petsc
 * @details Helper functions for working with DMPlex meshes.
 */

/**
 * @defgroup external_interfaces_petsc_dmplex_mesh_interface DMPlex Mesh Interface
 * @ingroup external_interfaces_petsc
 * @details Implementation which allows particles to exist on and interact with
 * DMPlex meshes.
 */

/**
 * @defgroup external_interfaces_petsc_dmplex_dsmc DMPlex DSMC
 * @ingroup external_interfaces_petsc
 * @details Implementation relating to DSMC simulations on DMPlex meshes.
 */

/**
 * @defgroup external_interfaces_petsc_dmplex_boundary_intersection DMPlex Boundary Intersection
 * @ingroup external_interfaces_petsc
 * @details Types and functions for detecting interactions between particle trajectories and the boundary of the DMPlex mesh.
 */

/**
 * @defgroup external_interfaces_petsc_dmplex_proj_eval DMPlex Function Projection and Evaluation
 * @ingroup external_interfaces_petsc
 * @details Implementation for projection (deposition) onto functions defined on DMPlex meshes and evaluating functions defined on DMPlex meshes. Please visit the boundary intersection section for methods that implement surface function interaction.
 */

/**
 * @defgroup external_interfaces_petsc_dmplex_mesh_coupling DMPlex Mesh Coupling
 * @ingroup external_interfaces_petsc
 * @details Types and functions for coupling DMPlex meshes with other meshes.
 */

// clang-format on

#ifdef NESO_PARTICLES_PETSC
#include "boundary_interaction/boundary_interaction.hpp"
#include "dmplex_interface.hpp"
#include "dmplex_local_mapper.hpp"
#include "dmplex_mesh_coupler_dg0.hpp"
#include "dmplex_project_evaluate.hpp"
#include "dmplex_utility.hpp"
#include "petsc_api_redirection.hpp"
#include "petsc_common.hpp"
#include "petsc_utility.hpp"
#include "project_evaluate/dmplex_function.hpp"
#include "project_evaluate/dmplex_function_mass_matrix.hpp"
#include "voronoi_cell_volume.hpp"
#endif

#endif
