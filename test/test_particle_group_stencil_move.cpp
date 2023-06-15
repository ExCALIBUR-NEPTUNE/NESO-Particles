#include <CL/sycl.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>
#include <vector>

using namespace NESO::Particles;

TEST(ParticleGroup, stencil_move_multiple) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 8;
  dims[1] = 4;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;

  auto mesh =
      std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                                       subdivision_order, stencil_width);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto domain = std::make_shared<Domain>(mesh);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  ParticleGroup A(domain, particle_spec, sycl_target);

  A.add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3),
                                 domain->mesh->get_cell_count()));

  // create object to map local cells + stencil to ranks
  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  // const int rank = sycl_target->comm_pair.rank_parent;
  // const int size = sycl_target->comm_pair.size_parent;

  // test the map in the mapper
  INT index_mesh[3];
  INT index_mh[6];

  auto k_lookup = cart_local_mapper->dh_lookup.h_buffer.ptr;
  auto k_lookup_stride = cart_local_mapper->lookup_stride;
  auto k_lookup_dims = cart_local_mapper->dh_lookup_dims.h_buffer.ptr;
  auto k_map = cart_local_mapper->dh_map.h_buffer.ptr;

  std::vector<int> covered_cells_x;
  std::vector<int> covered_cells_y;
  std::vector<int> covered_cells_z;

  for (int cx = mesh->cell_starts[0] - stencil_width;
       cx < mesh->cell_ends[0] + stencil_width; cx++) {
    int cx_wrapped =
        (cx + mesh->cell_counts[0] * stencil_width) % mesh->cell_counts[0];
    covered_cells_x.push_back(cx_wrapped);
  }
  for (int cx = mesh->cell_starts[1] - stencil_width;
       cx < mesh->cell_ends[1] + stencil_width; cx++) {
    int cx_wrapped = (cx + mesh->cell_counts[1] * stencil_width) %
                     MAX(mesh->cell_counts[1], 1);
    covered_cells_y.push_back(cx_wrapped);
  }
  for (int cx = mesh->cell_starts[2] - stencil_width;
       cx < mesh->cell_ends[2] + stencil_width; cx++) {
    int cx_wrapped = (cx + mesh->cell_counts[2] * stencil_width) %
                     MAX(mesh->cell_counts[2], 1);
    covered_cells_z.push_back(cx_wrapped);
  }

  std::vector<REAL> cell_centres[3];
  cell_centres[0].reserve(mesh->get_cell_count());
  cell_centres[1].reserve(mesh->get_cell_count());
  cell_centres[2].reserve(mesh->get_cell_count());

  int covered_cells = 0;
  const REAL cell_width = mesh->cell_width_fine;

  for (int cz = 0; cz < MAX(mesh->cell_counts[2], 1); cz++) {
    index_mesh[2] = cz;
    auto index_z = (ndim > 2) ? k_lookup[2 * k_lookup_stride + cz] : 0;

    for (int cy = 0; cy < MAX(mesh->cell_counts[1], 1); cy++) {
      index_mesh[1] = cy;
      auto index_y = (ndim > 1) ? k_lookup[k_lookup_stride + cy] : 0;

      for (int cx = 0; cx < mesh->cell_counts[0]; cx++) {
        index_mesh[0] = cx;
        auto index_x = k_lookup[cx];

        if (std::count(covered_cells_x.begin(), covered_cells_x.end(), cx) &&
            std::count(covered_cells_y.begin(), covered_cells_y.end(), cy) &&
            std::count(covered_cells_z.begin(), covered_cells_z.end(), cz)) {

          mesh->mesh_tuple_to_mh_tuple(index_mesh, index_mh);
          const INT index_global =
              mesh->get_mesh_hierarchy()->tuple_to_linear_global(index_mh);
          // get the rank that owns that global cell
          const int owning_rank =
              mesh->get_mesh_hierarchy()->get_owner(index_global);

          const int index_linear =
              index_x +
              k_lookup_dims[0] * (index_y + k_lookup_dims[1] * index_z);

          const int test_rank = k_map[index_linear];

          ASSERT_EQ(test_rank, owning_rank);
          cell_centres[0].push_back((cx + 0.5) * cell_width);
          cell_centres[1].push_back((cy + 0.5) * cell_width);
          cell_centres[2].push_back((cz + 0.5) * cell_width);

          covered_cells += 1;
        }
      }
    }
  }

  std::mt19937 rng_vel(52234231);

  const int N = covered_cells;
  const int Ntest = 1024;
  const REAL dt = 1.0;
  const REAL tol = 1.0e-10;
  const int cell_count = domain->mesh->get_cell_count();
  auto velocity_index = std::uniform_int_distribution<int>(0, 1);
  auto cell_index =
      std::uniform_int_distribution<int>(0, MAX(dims[0], dims[1]));

  ParticleSet initial_distribution(N, A.get_particle_spec());

  REAL vel_map[2] = {-1.0 / std::pow(2.0, subdivision_order),
                     1.0 / std::pow(2.0, subdivision_order)};
  for (int px = 0; px < N; px++) {

    // create particles in the centres of random cells
    for (int dimx = 0; dimx < ndim; dimx++) {
      const REAL pos = cell_centres[dimx][px];
      initial_distribution[Sym<REAL>("P")][px][dimx] = pos;
      initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = pos;
    }

    // initialise velocities that are +-1 cell in each direction
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] =
          vel_map[velocity_index(rng_vel)] * cell_extent;
    }

    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }

  A.add_particles_local(initial_distribution);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain->mesh);

  CartesianPeriodic pbc(sycl_target, mesh);
  CartesianCellBin ccb(sycl_target, mesh);

  reset_mpi_ranks(A[Sym<INT>("NESO_MPI_RANK")]);

  auto lambda_advect = [&] {
    auto k_P = A[Sym<REAL>("P")]->cell_dat.device_ptr();
    auto k_V = A[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_ndim = ndim;
    const auto k_dt = dt;

    const auto pl_iter_range = A.mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride = A.mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell = A.mpi_rank_dat->get_particle_loop_npart_cell();

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  k_P[cellx][dimx][layerx] += k_V[cellx][dimx][layerx] * k_dt;
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
  };

  REAL T = 0.0;

  int global_npart = 0;
  MPICHK(MPI_Allreduce(&N, &global_npart, 1, MPI_INT, MPI_SUM,
                       sycl_target->comm_pair.comm_parent));

  auto lambda_test = [&] {
    int npart_found = A.mpi_rank_dat->get_npart_local();
    int global_npart_found = 0;
    MPICHK(MPI_Allreduce(&npart_found, &global_npart_found, 1, MPI_INT, MPI_SUM,
                         sycl_target->comm_pair.comm_parent));
    ASSERT_EQ(global_npart_found, global_npart);

    // for all cells
    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto P = A[Sym<REAL>("P")]->cell_dat.get_cell(cellx);
      auto P_ORIG = A[Sym<REAL>("P_ORIG")]->cell_dat.get_cell(cellx);
      auto V = A[Sym<REAL>("V")]->cell_dat.get_cell(cellx);
      auto C = A[Sym<INT>("CELL_ID")]->cell_dat.get_cell(cellx);
      auto MPI_RANK = A[Sym<INT>("NESO_MPI_RANK")]->cell_dat.get_cell(cellx);
      auto ID = A[Sym<INT>("ID")]->cell_dat.get_cell(cellx);

      const int nrow = P->nrow;

      // for each dimension
      for (int dimx = 0; dimx < ndim; dimx++) {

        // for each particle
        for (int px = 0; px < nrow; px++) {
          // read the original position of the particle and compute the correct
          // current position based on the time T and velocity on the particle
          const REAL P_correct_abs = (*P_ORIG)[dimx][px] + T * (*V)[dimx][px];
          // map the absolute position back into the periodic domain

          const REAL extent = mesh->global_extents[dimx];
          const REAL P_correct =
              std::fmod(std::fmod(P_correct_abs, extent) + extent, extent);

          const REAL P_to_test = (*P)[dimx][px];

          const REAL err0 = ABS(P_correct - P_to_test);
          // case where P_correct is at 0 and P_to_test is at extent - which is
          // the same point in the periodic mapping
          const REAL err1 = ABS(err0 - extent);

          ASSERT_TRUE(((err0 <= tol) || (err1 <= tol)));

          // check that the particle position is actually owned by this MPI
          // rank

          auto mpi_rank_0 = (*MPI_RANK)[0][px];
          auto mpi_rank_1 = (*MPI_RANK)[1][px];

          const INT id = (*ID)[0][px];
          const int particle_cell =
              ((REAL)(P_to_test * mesh->inverse_cell_width_fine));

          ASSERT_TRUE(particle_cell >= mesh->cell_starts[dimx]);
          ASSERT_TRUE(particle_cell < mesh->cell_ends[dimx]);
        }
      }

      // check the particle is in the cell it is binned into
      // for each particle
      for (int px = 0; px < nrow; px++) {
        int index_tuple[3] = {0, 0, 0};
        for (int dimx = 0; dimx < ndim; dimx++) {
          const REAL P_CELL =
              (*P)[dimx][px] - mesh->cell_starts[dimx] * mesh->cell_width_fine;
          index_tuple[dimx] = ((REAL)P_CELL * mesh->inverse_cell_width_fine);
        }
        int index_linear =
            index_tuple[0] +
            mesh->cell_counts_local[0] *
                (index_tuple[1] + mesh->cell_counts_local[1] * index_tuple[2]);

        ASSERT_EQ((*C)[0][px], index_linear);
      }
    }
  };

  auto lambda_test_mapping = [&] {
    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto P = A[Sym<REAL>("P")]->cell_dat.get_cell(cellx);
      auto C = A[Sym<INT>("CELL_ID")]->cell_dat.get_cell(cellx);
      auto MPI_RANK = A[Sym<INT>("NESO_MPI_RANK")]->cell_dat.get_cell(cellx);

      const int nrow = P->nrow;

      // for each particle
      for (int px = 0; px < nrow; px++) {
        const int rank_global = (*MPI_RANK)[0][px];
        const int rank_local = (*MPI_RANK)[1][px];

        if (rank_local >= 0) {
          ASSERT_EQ(rank_global, rank_local);
        }
      }
    }
  };

  reset_mpi_ranks(A.mpi_rank_dat);
  mesh_hierarchy_global_map.execute(A.position_dat, A.cell_id_dat, A.mpi_rank_dat);
  cart_local_mapper->map(A);
  lambda_test_mapping();

  for (int testx = 0; testx < Ntest; testx++) {
    pbc.execute(A.position_dat);

    reset_mpi_ranks(A.mpi_rank_dat);
    mesh_hierarchy_global_map.execute(A.position_dat, A.cell_id_dat, A.mpi_rank_dat);

    cart_local_mapper->map(A);
    lambda_test_mapping();

    A.local_move();

    ccb.execute(A.position_dat, A.cell_id_dat);

    A.cell_move();

    lambda_test();
    lambda_advect();

    T += dt;
  }

  mesh->free();
}
