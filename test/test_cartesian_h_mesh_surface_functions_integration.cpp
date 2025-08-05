#include "include/test_neso_particles.hpp"

namespace {

void mass_conservation_wrapper(const int ndim) {
  const int nx = 16;
  const int ny = 32;
  const int nz = 6;
  const int npart_cell = 4;
  const REAL dt = 0.20;
  const int Nsteps = 100;

  auto [A_t, sycl_target, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);

  auto A = A_t;
  A->add_particle_dat(Sym<REAL>("MASS"), 1);

  particle_loop(
      A, [=](auto MASS) { MASS.at(0) = 1.0; }, Access::write(Sym<REAL>("MASS")))
      ->execute();

  auto mesh = std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh);

  std::map<int, std::vector<int>> boundary_groups;
  if (ndim == 2) {
    boundary_groups[0] = {0, 1, 2, 3};
  } else {
    boundary_groups[0] = {0, 1, 2, 3, 4, 5};
  }

  auto cti = std::make_shared<CartesianTrajectoryIntersection>(
      sycl_target, mesh, boundary_groups, 1.0e-14);
  cti->prepare_particle_group(A);

  auto func_mass = cti->create_function(0, "DG", 0);

  auto lambda_get_boundary_mass = [&]() -> REAL {
    REAL local_mass = 0.0;
    auto h_dofs = func_mass->get_dofs();
    for (auto ix : h_dofs) {
      local_mass += ix;
    }
    REAL total_mass = 0.0;
    MPICHK(MPI_Allreduce(&local_mass, &total_mass, 1,
                         map_ctype_mpi_type<REAL>(), MPI_SUM,
                         mesh->get_comm()));

    return total_mass * std::pow(mesh->cell_width_fine, ndim - 1);
  };

  auto advection_loop = particle_loop(
      A,
      [=](auto P, auto V) {
        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) += dt * V.at(dx);
        }
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")));

  auto ga_mass = std::make_shared<GlobalArray<REAL>>(sycl_target, 1);
  auto lambda_get_particle_mass = [&]() -> REAL {
    ga_mass->fill(0.0);
    particle_loop(
        A, [=](auto MASS, auto GA_MASS) { GA_MASS.add(0, MASS.at(0)); },
        Access::read(Sym<REAL>("MASS")), Access::add(ga_mass))
        ->execute();
    return ga_mass->get().at(0);
  };

  // auto h5part = std::make_shared<H5Part>(
  //     "trajectory_surface_" + std::to_string(ndim) + ".h5part", A,
  //     Sym<REAL>("P"));

  REAL mass_particle_system = lambda_get_particle_mass();
  const REAL mass_total = mass_particle_system;
  REAL mass_boundary_function = lambda_get_boundary_mass();

  for (int stepx = 0; stepx < Nsteps; stepx++) {
    cti->pre_integration(A);
    advection_loop->execute();
    auto groups = cti->post_integration(A);

    cti->function_project(groups[0], Sym<REAL>("MASS"), 0, false, func_mass);
    if (groups.count(0)) {
      A->remove_particles(groups.at(0));
    }

    mass_particle_system = lambda_get_particle_mass();
    mass_boundary_function = lambda_get_boundary_mass();
    const REAL mass_diff =
        mass_total - mass_boundary_function - mass_particle_system;
    ASSERT_NEAR(mass_diff, 0.0, 1.0e-10);

    // func_mass->write_vtkhdf("trajectory_surface_" + std::to_string(ndim) +
    //                         "d_" + std::to_string(stepx) + ".vtkhdf");

    // h5part->write();
  }

  // h5part->close();

  cti->free();
  mesh->free();
}

} // namespace

TEST(CartesianHMesh, surface_functions_mass_conservation_2d) {
  mass_conservation_wrapper(2);
}

TEST(CartesianHMesh, surface_functions_mass_conservation_3d) {
  mass_conservation_wrapper(3);
}
