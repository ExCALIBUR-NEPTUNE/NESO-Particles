#include "include/test_neso_particles.hpp"

TEST(ParticleIO, h5_part_write_particle_group) {

#ifdef NESO_PARTICLES_HDF5
  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 8;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 0;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  // create object to map local cells + stencil to ranks
  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true),
      ParticleProp(Sym<REAL>("V"), 3),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("ID"), 1),
      ParticleProp(Sym<INT>("ID2"), 1),
  };

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  CartesianPeriodic pbc(sycl_target, mesh, A->position_dat);
  CartesianCellBin ccb(sycl_target, mesh, A->position_dat, A->cell_id_dat);

  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  H5Part h5parte("test_empty_dump.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"),
                 Sym<INT>("ID"), Sym<INT>("ID2"), Sym<INT>("NESO_MPI_RANK"));

  h5parte.write();
  h5parte.close();

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);
  std::mt19937 rng_rank(18241);

  const int N = 1024;

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  std::uniform_int_distribution<int> uniform_dist(
      0, sycl_target->comm_pair.size_parent - 1);

  ParticleSet initial_distribution(N, A->get_particle_spec());

  // determine which particles should end up on which rank
  std::map<int, std::vector<int>> mapping;
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
    initial_distribution[Sym<INT>("ID2")][px][0] = px * 10;
    const auto px_rank = uniform_dist(rng_rank);
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
    mapping[px_rank].push_back(px);
  }

  if (sycl_target->comm_pair.rank_parent == 0) {
    A->add_particles_local(initial_distribution);
  }

  A->hybrid_move();
  ccb.execute();
  A->cell_move();

  H5Part h5part("test_dump.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"),
                Sym<INT>("ID"), Sym<INT>("ID2"), Sym<INT>("NESO_MPI_RANK"));

  h5part.write();
  h5part.close();

  if (sycl_target->comm_pair.rank_parent == 0) {
    std::vector<long long> data_ll(N);
    std::vector<double> data_real(N);
    std::vector<long long> ordering(N);

    hid_t file_id = H5Fopen("test_dump.h5part", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t group_step = H5Gopen(file_id, "Step#0", H5P_DEFAULT);

    hid_t dataset = H5Dopen(group_step, "ID_0", H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dataset);
    hsize_t d_rank = H5Sget_simple_extent_ndims(filespace);
    ASSERT_EQ(d_rank, 1);

    hsize_t dims[1];
    H5CHK(H5Sget_simple_extent_dims(filespace, dims, NULL));
    ASSERT_EQ(dims[0], N);

    hid_t memspace = H5Screate_simple(1, dims, NULL);

    H5CHK(H5Dread(dataset, H5T_NATIVE_LLONG, memspace, filespace, H5P_DEFAULT,
                  data_ll.data()));

    for (int px = 0; px < N; px++) {
      const long long tmp_id = data_ll[px];
      ordering[px] = tmp_id;
    }

    H5CHK(H5Sclose(memspace));
    H5CHK(H5Sclose(filespace));
    H5CHK(H5Dclose(dataset));

    auto get_real = [&](std::string attr_name) {
      hid_t dataset = H5Dopen(group_step, attr_name.c_str(), H5P_DEFAULT);
      hid_t filespace = H5Dget_space(dataset);
      hsize_t d_rank = H5Sget_simple_extent_ndims(filespace);
      ASSERT_EQ(d_rank, 1);
      hsize_t dims[1];
      H5CHK(H5Sget_simple_extent_dims(filespace, dims, NULL));
      ASSERT_EQ(dims[0], N);
      hid_t memspace = H5Screate_simple(1, dims, NULL);
      H5CHK(H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, filespace,
                    H5P_DEFAULT, data_real.data()));
      H5CHK(H5Sclose(memspace));
      H5CHK(H5Sclose(filespace));
      H5CHK(H5Dclose(dataset));
    };

    for (int dimx = 0; dimx < 2; dimx++) {
      get_real("P_" + std::to_string(dimx));
      for (int px = 0; px < N; px++) {
        const int orig_px = ordering[px];
        ASSERT_EQ(data_real[px],
                  initial_distribution[Sym<REAL>("P")][orig_px][dimx]);
      }
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      get_real("V_" + std::to_string(dimx));
      for (int px = 0; px < N; px++) {
        const int orig_px = ordering[px];
        ASSERT_EQ(data_real[px],
                  initial_distribution[Sym<REAL>("V")][orig_px][dimx]);
      }
    }

    get_real("x");
    for (int px = 0; px < N; px++) {
      const int orig_px = ordering[px];
      ASSERT_EQ(data_real[px],
                initial_distribution[Sym<REAL>("P")][orig_px][0]);
    }
    get_real("y");
    for (int px = 0; px < N; px++) {
      const int orig_px = ordering[px];
      ASSERT_EQ(data_real[px],
                initial_distribution[Sym<REAL>("P")][orig_px][1]);
    }

    auto get_int = [&](std::string attr_name) {
      hid_t dataset = H5Dopen(group_step, attr_name.c_str(), H5P_DEFAULT);
      hid_t filespace = H5Dget_space(dataset);
      hsize_t d_rank = H5Sget_simple_extent_ndims(filespace);
      ASSERT_EQ(d_rank, 1);
      hsize_t dims[1];
      H5CHK(H5Sget_simple_extent_dims(filespace, dims, NULL));
      ASSERT_EQ(dims[0], N);
      hid_t memspace = H5Screate_simple(1, dims, NULL);
      H5CHK(H5Dread(dataset, H5T_NATIVE_LLONG, memspace, filespace, H5P_DEFAULT,
                    data_ll.data()));
      H5CHK(H5Sclose(memspace));
      H5CHK(H5Sclose(filespace));
      H5CHK(H5Dclose(dataset));
    };

    get_int("ID2_0");
    for (int px = 0; px < N; px++) {
      const int orig_px = ordering[px];
      ASSERT_EQ(data_ll[px], initial_distribution[Sym<INT>("ID2")][orig_px][0]);
    }

    H5CHK(H5Gclose(group_step));
    H5CHK(H5Fclose(file_id));
  }

  {
    A->clear();

    H5Part h5part("test_dump_empty.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"),
                  Sym<INT>("ID"), Sym<INT>("ID2"), Sym<INT>("NESO_MPI_RANK"));
    h5part.write();
    h5part.close();
  }

  sycl_target->free();
  mesh->free();

#endif
}

TEST(ParticleIO, h5_part_read_particle_group) {
#ifdef NESO_PARTICLES_HDF5
  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 8;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 0;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  // create object to map local cells + stencil to ranks
  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true),
      ParticleProp(Sym<REAL>("P2"), ndim),
      ParticleProp(Sym<REAL>("V"), 3),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("ID"), 1),
      ParticleProp(Sym<INT>("ID2"), 5),
  };

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  CartesianPeriodic pbc(sycl_target, mesh, A->position_dat);
  CartesianCellBin ccb(sycl_target, mesh, A->position_dat, A->cell_id_dat);

  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  H5Part h5parte("test_empty_dump.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"),
                 Sym<INT>("ID"), Sym<INT>("ID2"), Sym<INT>("NESO_MPI_RANK"));

  h5parte.write();
  h5parte.close();

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);
  std::mt19937 rng_rank(18241);

  const int N = 1024;

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  std::uniform_int_distribution<int> uniform_dist(
      0, sycl_target->comm_pair.size_parent - 1);

  ParticleSet initial_distribution(N, A->get_particle_spec());

  for (int dimx = 0; dimx < ndim; dimx++) {
    initial_distribution.set(Sym<REAL>("P"), dimx, positions[dimx]);
  }
  for (int dimx = 0; dimx < 3; dimx++) {
    initial_distribution.set(Sym<REAL>("V"), dimx, velocities[dimx]);
  }

  // determine which particles should end up on which rank
  std::map<int, std::vector<int>> mapping;
  for (int px = 0; px < N; px++) {
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;

    for (int ix = 0; ix < 5; ix++) {
      initial_distribution[Sym<INT>("ID2")][px][ix] = (px * 10 + ix) % 7;
    }

    const auto px_rank = uniform_dist(rng_rank);
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
    mapping[px_rank].push_back(px);
  }

  if (sycl_target->comm_pair.rank_parent == 0) {
    A->add_particles_local(initial_distribution);
  }

  A->hybrid_move();
  ccb.execute();
  A->cell_move();

  {
    H5Part h5part("test_dump.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"),
                  Sym<INT>("ID"), Sym<INT>("ID2"), Sym<INT>("NESO_MPI_RANK"));

    h5part.write();
    h5part.close();
  }

  {
    ParticleSpec particle_spec_read(
        ParticleProp(Sym<REAL>("P2"), ndim, true),
        ParticleProp(Sym<REAL>("P"), ndim), ParticleProp(Sym<REAL>("V"), 3),
        ParticleProp(Sym<INT>("ID"), 1), ParticleProp(Sym<INT>("ID2"), 5));
    H5Part h5part("test_dump.h5part", sycl_target);
    auto particle_set = h5part.read(particle_spec_read, 0, true);
    h5part.close();

    auto B = std::make_shared<ParticleGroup>(A->domain, A->get_particle_spec(),
                                             A->sycl_target);

    ParticleSet particle_set_b(particle_set->npart, B->get_particle_spec());
    particle_set_b.set(particle_set);

    B->add_particles_local(particle_set_b);
    parallel_advection_initialisation(B, 16);

    CartesianCellBin ccbB(sycl_target, mesh, B->position_dat, B->cell_id_dat);
    ccbB.execute();
    B->cell_move();

    auto la_int = std::make_shared<LocalArray<INT>>(
        sycl_target, N * particle_spec_read.get_max_ncomp_int());
    auto la_real = std::make_shared<LocalArray<REAL>>(
        sycl_target, N * particle_spec_read.get_max_ncomp_real());

    particle_loop(
        A,
        [=](auto P, auto P2) {
          for (int dx = 0; dx < ndim; dx++) {
            P2.at(dx) = P.at(dx);
          }
        },
        Access::read(Sym<REAL>("P")), Access::write(Sym<REAL>("P2")))
        ->execute();

    auto lambda_test_wrapper = [&](auto px, auto la) {
      la->fill(-10000);

      auto ep = ErrorPropagate(sycl_target);
      auto k_ep = ep.device_ptr();
      auto sym = px.sym;
      const int ncomp = px.ncomp;
      particle_loop(
          A,
          [=](auto ID, auto DAT, auto LA) {
            const INT index = ID.at(0);
            const bool id_is_good = (0 <= index) && (index < N);
            NESO_KERNEL_ASSERT(id_is_good, k_ep);
            if (id_is_good) {
              for (int cx = 0; cx < ncomp; cx++) {
                LA.at(index * ncomp + cx) = DAT.at(cx);
              }
            }
          },
          Access::read(Sym<INT>("ID")), Access::read(sym), Access::write(la))
          ->execute();
      ASSERT_FALSE(ep.get_flag());
      particle_loop(
          B,
          [=](auto ID, auto DAT, auto LA) {
            const INT index = ID.at(0);
            const bool id_is_good = (0 <= index) && (index < N);
            NESO_KERNEL_ASSERT(id_is_good, k_ep);
            if (id_is_good) {
              for (int cx = 0; cx < ncomp; cx++) {
                const auto correct = LA.at(index * ncomp + cx);
                const auto to_test = DAT.at(cx);
                NESO_KERNEL_ASSERT(correct == to_test, k_ep);
              }
            }
          },
          Access::read(Sym<INT>("ID")), Access::read(sym), Access::read(la))
          ->execute();
      ASSERT_FALSE(ep.get_flag());
    };

    for (auto &px : particle_spec_read.properties_int) {
      lambda_test_wrapper(px, la_int);
    }
    for (auto &px : particle_spec_read.properties_real) {
      lambda_test_wrapper(px, la_real);
    }
  }

  sycl_target->free();
  mesh->free();
#endif
}
