#include "include/test_neso_particles.hpp"

TEST(CartesianHMesh, surface_functions_2d) {
  const int ndim = 2;
  const REAL dt = 1000.0;

  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);
  auto mesh = std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh);

  nprint("rank:", sycl_target->comm_pair.rank_parent,
         "owner:", mesh->get_face_id_owning_rank(0));

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {0, 1, 2};
  boundary_groups[1] = {3};

  CartesianTrajectoryIntersection cti(sycl_target, mesh, boundary_groups,
                                      1.0e-14);

  auto u0 = cti.create_function(0, "DG", 0);

  const std::size_t num_geoms = mesh->get_all_face_cells_on_face(0).size() +
                                mesh->get_all_face_cells_on_face(1).size() +
                                mesh->get_all_face_cells_on_face(2).size();

  ASSERT_EQ(u0->d_dofs->size, num_geoms);
  ASSERT_EQ(u0->local_dof_count, num_geoms);
  ASSERT_EQ(u0->cell_dof_count, 1);
  ASSERT_EQ(u0->mesh, mesh);
  ASSERT_EQ(u0->sycl_target, sycl_target);
  ASSERT_EQ(u0->ndim, 1);
  ASSERT_EQ(u0->cell_count, num_geoms);
  ASSERT_EQ(u0->function_space, "DG");
  ASSERT_EQ(u0->polynomial_order, 0);
  ASSERT_EQ(u0->cells.size(), num_geoms);
  ASSERT_EQ(u0->element_group, 0);

  // Move particles to the boundary
  cti.prepare_particle_group(A);
  cti.pre_integration(A);

  particle_loop(
      A,
      [=](auto V, auto P) {
        REAL r2 = 0.0;
        for (int dx = 0; dx < ndim; dx++) {
          r2 += V.at(dx) * V.at(dx);
        }
        if (r2 > 0.0) {
          const REAL ir = 1.0 / Kernel::sqrt(r2);
          for (int dx = 1; dx < ndim; dx++) {
            V.at(dx) = V.at(dx) * ir;
          }
        } else {
          V.at(0) = 1.0;
          for (int dx = 1; dx < ndim; dx++) {
            V.at(dx) = 0.0;
          }
        }

        // update the positions
        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) += dt * V.at(dx);
        }
      },
      Access::write(Sym<REAL>("V")), Access::write(Sym<REAL>("P")))
      ->execute();

  A->add_particle_dat(Sym<REAL>("Q"), 2);
  std::mt19937 rng_state(52234234);
  std::normal_distribution<REAL> rng_dist(0, 1.0);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng_state); };
  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, 2);

  particle_loop(
      A,
      [=](auto INDEX, auto Q, auto RNG) {
        Q.at(0) = RNG.at(INDEX, 0);
        Q.at(1) = RNG.at(INDEX, 1);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("Q")),
      Access::read(rng_kernel))
      ->execute();

  const auto ncells_face_global = mesh->ncells_face_global;
  auto func_ga =
      std::make_shared<GlobalArray<REAL>>(sycl_target, ncells_face_global);
  func_ga->fill(0.0);
  auto groups = cti.post_integration(A);

  // Check all the particles made it to the wall.
  ASSERT_EQ(A->get_npart_local(),
            groups.at(0)->get_npart_local() + groups.at(1)->get_npart_local());

  for (auto &gx : groups) {
    const REAL scaling = mesh->inverse_cell_width_fine;
    ErrorPropagate ep(sycl_target);
    auto k_ep = ep.device_ptr();
    particle_loop(
        gx.second,
        [=](auto META_EPH, auto Q, auto FUNC_GA) {
          const INT geom_id = META_EPH.at_ephemeral(1);
          const bool valid_geom_id =
              (-1 < geom_id) && (geom_id < ncells_face_global);
          NESO_KERNEL_ASSERT(valid_geom_id, k_ep);
          if (valid_geom_id) {
            const REAL value = scaling * Q.at(0);
            FUNC_GA.add(geom_id, value);
          }
        },
        Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
        Access::read(Sym<REAL>("Q")), Access::add(func_ga))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  }

  // Test some of the device maps
  for (int gx : {0, 1}) {
    auto bmi = cti.map_groups_boundary_interface.at(gx);

    const std::size_t num_incoming_geoms =
        bmi->boundary.d_incoming_geom_ids->size;

    if (num_incoming_geoms > 0) {
      int *k_incoming_geom_ids = bmi->boundary.d_incoming_geom_ids->ptr;
      auto *k_map_owned_geom_id_to_linear_index =
          bmi->boundary.d_map_owned_geom_id_to_linear_index->root;

      ASSERT_NE(k_map_owned_geom_id_to_linear_index, nullptr);

      ErrorPropagate ep(sycl_target);
      auto k_ep = ep.device_ptr();

      sycl_target->queue
          .parallel_for(sycl::range<1>(num_incoming_geoms),
                        [=](auto idx) {
                          const int gid = k_incoming_geom_ids[idx];
                          const INT *linear_index = nullptr;
                          const bool found =
                              k_map_owned_geom_id_to_linear_index->get(
                                  gid, &linear_index);
                          NESO_KERNEL_ASSERT(found, k_ep);
                        })
          .wait_and_throw();
      ASSERT_FALSE(ep.get_flag());
    }
  }

  auto func_0 = cti.create_function(0, "DG", 0);
  auto func_1 = cti.create_function(1, "DG", 0);

  cti.function_project(groups.at(0), Sym<REAL>("Q"), 0, false, func_0);
  cti.function_project(groups.at(1), Sym<REAL>("Q"), 0, false, func_1);

  std::vector<std::pair<CartesianHMeshFunctionSharedPtr,
                        BoundaryMeshInterfaceSharedPtr>>
      rgroups = {{func_0, cti.map_groups_boundary_interface.at(0)},
                 {func_1, cti.map_groups_boundary_interface.at(1)}};

  auto h_correct_dofs = func_ga->get();

  for (auto &func_bmi : rgroups) {
    auto func = func_bmi.first;
    auto bmi = func_bmi.second;

    std::vector<REAL> h_dofs(func->local_dof_count);
    sycl_target->queue
        .memcpy(h_dofs.data(), func->d_dofs->ptr,
                func->local_dof_count * sizeof(REAL))
        .wait_and_throw();

    int index = 0;
    for (auto gidx : bmi->boundary.owned_geom_ids) {
      const REAL to_test = h_dofs.at(index);
      const REAL correct = h_correct_dofs.at(gidx);
      nprint("gid:", gidx, "error:", relative_error(correct, to_test), correct,
             to_test);
      // ASSERT_TRUE(relative_error(correct, to_test) < 1.0e-12);
      index++;
    }
  }

  cti.free();
  mesh->free();
}
