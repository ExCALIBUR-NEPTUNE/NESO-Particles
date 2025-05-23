#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>

namespace {
class TestBoundaryInteraction2D : public PetscInterface::BoundaryInteraction2D {
public:
  TestBoundaryInteraction2D(
      SYCLTargetSharedPtr sycl_target,
      PetscInterface::DMPlexInterfaceSharedPtr mesh,
      std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
      const REAL tol = 0.0,
      std::optional<Sym<REAL>> previous_position_sym = std::nullopt)
      :

        PetscInterface::BoundaryInteraction2D(
            sycl_target, mesh, boundary_groups, tol, previous_position_sym) {}

  using BoundaryInteractionCellData2D =
      PetscInterface::BoundaryInteractionCellData2D;
  using BoundaryInteractionNormalData2D =
      PetscInterface::BoundaryInteractionNormalData2D;

  MAKE_WRAP_METHOD(get_labels)
  MAKE_WRAP_METHOD(get_bounding_box)
  MAKE_WRAP_METHOD(collect_cells)
  MAKE_GETTER_METHOD(facets_real)
  MAKE_GETTER_METHOD(facets_int)
  MAKE_GETTER_METHOD(num_facets_global)
  MAKE_GETTER_METHOD(ncomp_int)
  MAKE_GETTER_METHOD(ncomp_real)
  MAKE_GETTER_METHOD(required_mh_cells)
  MAKE_GETTER_METHOD(collected_mh_cells)
  MAKE_GETTER_METHOD(d_map_edge_discovery)
  MAKE_GETTER_METHOD(d_map_edge_normals)
};

ParticleGroupSharedPtr particle_loop_common(DM dm, const int N = 1093) {

  const int ndim = 2;
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("U"), 3),
                             ParticleProp(Sym<REAL>("TSP"), 2),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 2),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  int ncell_local = mesh->get_cell_count();
  int ncell_global;

  MPICHK(MPI_Allreduce(&ncell_local, &ncell_global, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD));
  const int npart_per_cell = std::max(1, N / ncell_global);
  const int rank = sycl_target->comm_pair.rank_parent;
  const INT id_offset = rank * N;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::vector<std::vector<double>> positions;
  std::vector<int> cells;

  uniform_within_dmplex_cells(mesh, npart_per_cell, positions, cells, &rng_pos);

  const int N_actual = cells.size();
  auto velocities =
      NESO::Particles::normal_distribution(N_actual, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N_actual, particle_spec);

  for (int px = 0; px < N_actual; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<INT>("ID")][px][0] = px + id_offset;
  }

  A->add_particles_local(initial_distribution);

  return A;
}

} // namespace

TEST(PETScBoundary2D, reflection_truncated) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int nsteps = 200;
  const REAL dt = 0.1;

  const int ndim = 2;
  const int mesh_size = 8;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces, lower, upper,
      /* periodicity */ NULL, PETSC_TRUE, &dm));
  PetscInterface::generic_distribute(&dm);
  auto A = particle_loop_common(dm, 1024);
  auto mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
      A->domain->mesh);
  auto sycl_target = A->sycl_target;

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[0] = {1};
  boundary_groups[1] = {2};
  boundary_groups[2] = {3};
  boundary_groups[3] = {4};

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);
  auto reflection =
      std::make_shared<ExternalCommon::BoundaryReflection>(2, 1.0e-10);

  auto ep = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = ep->device_ptr();

  auto lambda_apply_boundary_conditions = [&](auto aa) {
    auto sub_groups = b2d->post_integration(aa);

    for (auto &gx : sub_groups) {
      particle_loop(
          gx.second,
          [=](auto V, auto U) {
            U.at(0) = V.at(0);
            U.at(1) = V.at(1);
            U.at(2) = V.at(2);
          },
          Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("U")))
          ->execute();

      reflection->execute(gx.second, Sym<REAL>("P"), Sym<REAL>("V"),
                          Sym<REAL>("TSP"), b2d->previous_position_sym);

      particle_loop(
          gx.second,
          [=](auto V, auto U) {
            const REAL V_mag =
                V.at(0) * V.at(0) + V.at(1) * V.at(1) + V.at(2) * V.at(2);
            const REAL U_mag =
                U.at(0) * U.at(0) + U.at(1) * U.at(1) + U.at(2) * U.at(2);

            NESO_KERNEL_ASSERT(Kernel::abs(V_mag - U_mag) < 1.0e-14, k_ep);

            if ((Kernel::abs(V.at(0)) > 1.0e-14) &&
                (Kernel::abs(V.at(1)) > 1.0e-14)) {
              const bool x_flipped = Kernel::abs(V.at(0) + U.at(0)) < 1.0e-15;
              const bool y_flipped = Kernel::abs(V.at(1) + U.at(1)) < 1.0e-15;
              NESO_KERNEL_ASSERT(x_flipped != y_flipped, k_ep);
              if (x_flipped) {
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(1) - U.at(1)) < 1.0e-15,
                                   k_ep);
              }
              if (y_flipped) {
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(0) - U.at(0)) < 1.0e-15,
                                   k_ep);
              }
            }
          },
          Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("U")))
          ->execute();
      ASSERT_EQ(ep->get_flag(), 0);

      particle_loop(
          gx.second,
          [=](auto P) {
            NESO_KERNEL_ASSERT((0.0 < P.at(0)) && (P.at(0) < mesh_size * h),
                               k_ep);
            NESO_KERNEL_ASSERT((0.0 < P.at(1)) && (P.at(1) < mesh_size * h),
                               k_ep);
          },
          Access::read(Sym<REAL>("P")))
          ->execute();
      ASSERT_EQ(ep->get_flag(), 0);
    }
  };

  auto lambda_apply_timestep_reset = [&](auto aa) {
    particle_loop(
        aa,
        [=](auto TSP) {
          TSP.at(0) = 0.0;
          TSP.at(1) = 0.0;
        },
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };
  auto lambda_apply_advection_step =
      [=](ParticleSubGroupSharedPtr iteration_set) -> void {
    particle_loop(
        "euler_advection", iteration_set,
        [=](auto V, auto P, auto TSP) {
          const REAL dt_left = dt - TSP.at(0);
          if (dt_left > 0.0) {
            P.at(0) += dt_left * V.at(0);
            P.at(1) += dt_left * V.at(1);
            TSP.at(0) = dt;
            TSP.at(1) = dt_left;
          }
        },
        Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("P")),
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };
  auto lambda_pre_advection = [&](auto aa) { b2d->pre_integration(aa); };
  auto lambda_find_partial_moves = [&](auto aa) {
    return static_particle_sub_group(
        aa, [=](auto TSP) { return TSP.at(0) < dt; },
        Access::read(Sym<REAL>("TSP")));
  };
  auto lambda_partial_moves_remaining = [&](auto aa) -> bool {
    const int size = aa->get_npart_local();
    return size > 0;
  };
  auto lambda_apply_timestep = [&](auto aa) {
    lambda_apply_timestep_reset(aa);
    lambda_pre_advection(aa);
    lambda_apply_advection_step(aa);
    lambda_apply_boundary_conditions(aa);
    aa = lambda_find_partial_moves(aa);
    while (lambda_partial_moves_remaining(aa)) {
      lambda_pre_advection(aa);
      lambda_apply_advection_step(aa);
      lambda_apply_boundary_conditions(aa);
      aa = lambda_find_partial_moves(aa);
    }
  };

  for (int stepx = 0; stepx < nsteps; stepx++) {
    lambda_apply_timestep(static_particle_sub_group(A));
    A->hybrid_move();
    A->cell_move();
  }

  b2d->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary2D, reflection_advection) {
  PETSCCHK(PetscInitializeNoArguments());
  std::string mesh_file;
  GET_TEST_RESOURCE(mesh_file, "mesh_ring/mesh_ring.msh");
  if (mesh_file.size()) {

    DM dm;

    const int nsteps = 500;
    const REAL dt = 0.01;

    PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD, mesh_file.c_str(),
                                      (PetscBool)1, &dm));

    PetscInterface::generic_distribute(&dm);
    auto A = particle_loop_common(dm, 2048);
    auto mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
        A->domain->mesh);
    auto sycl_target = A->sycl_target;

    std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
    boundary_groups[1] = {1, 2, 3, 4};

    auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                           boundary_groups);
    auto reflection =
        std::make_shared<ExternalCommon::BoundaryReflection>(2, 1.0e-10);

    auto lambda_apply_boundary_conditions = [&](auto aa) {
      auto sub_groups = b2d->post_integration(aa);
      for (auto &gx : sub_groups) {
        reflection->execute(gx.second, Sym<REAL>("P"), Sym<REAL>("V"),
                            Sym<REAL>("TSP"), b2d->previous_position_sym);
      }
    };
    auto lambda_apply_timestep_reset = [&](auto aa) {
      particle_loop(
          aa,
          [=](auto TSP) {
            TSP.at(0) = 0.0;
            TSP.at(1) = 0.0;
          },
          Access::write(Sym<REAL>("TSP")))
          ->execute();
    };
    auto lambda_apply_advection_step =
        [=](ParticleSubGroupSharedPtr iteration_set) -> void {
      particle_loop(
          "euler_advection", iteration_set,
          [=](auto V, auto P, auto TSP) {
            const REAL dt_left = dt - TSP.at(0);
            if (dt_left > 0.0) {
              P.at(0) += dt_left * V.at(0);
              P.at(1) += dt_left * V.at(1);
              TSP.at(0) = dt;
              TSP.at(1) = dt_left;
            }
          },
          Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("P")),
          Access::write(Sym<REAL>("TSP")))
          ->execute();
    };
    auto lambda_pre_advection = [&](auto aa) { b2d->pre_integration(aa); };
    auto lambda_find_partial_moves = [&](auto aa) {
      return static_particle_sub_group(
          aa, [=](auto TSP) { return TSP.at(0) < dt; },
          Access::read(Sym<REAL>("TSP")));
    };
    auto lambda_partial_moves_remaining = [&](auto aa) -> bool {
      const int size = aa->get_npart_local();
      return size > 0;
    };
    auto lambda_apply_timestep = [&](auto aa) {
      lambda_apply_timestep_reset(aa);
      lambda_pre_advection(aa);
      lambda_apply_advection_step(aa);
      lambda_apply_boundary_conditions(aa);
      aa = lambda_find_partial_moves(aa);
      while (lambda_partial_moves_remaining(aa)) {
        lambda_pre_advection(aa);
        lambda_apply_advection_step(aa);
        lambda_apply_boundary_conditions(aa);
        aa = lambda_find_partial_moves(aa);
      }
    };

    // H5Part h5part("traj_reflection.h5part", A, Sym<REAL>("P"),
    // Sym<REAL>("V"));

    for (int stepx = 0; stepx < nsteps; stepx++) {
      // if ((stepx % 20 == 0) && (sycl_target->comm_pair.rank_parent == 0)) {
      //   nprint("step:", stepx);
      // }
      lambda_apply_timestep(static_particle_sub_group(A));
      A->hybrid_move();
      A->cell_move();
      // h5part.write();
    }
    // h5part.close();

    b2d->free();
    sycl_target->free();
    mesh->free();
    PETSCCHK(DMDestroy(&dm));
  }
  PETSCCHK(PetscFinalize());
}

#endif
