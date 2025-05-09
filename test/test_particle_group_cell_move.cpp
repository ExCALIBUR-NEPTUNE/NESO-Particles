#include "include/test_neso_particles.hpp"

TEST(ParticleGroup, cell_move) {

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
  auto domain = std::make_shared<Domain>(mesh);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = make_test_obj<ParticleGroup>(domain, particle_spec, sycl_target);

  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int size = sycl_target->comm_pair.size_parent;
  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_cell(18241 + rank);
  std::mt19937 rng_rank(112348241);

  const int N = 1024;
  const int Ntest = 20;

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);

  const int cell_count = domain->mesh->get_cell_count();
  std::uniform_int_distribution<int> dist_cell(0, cell_count - 1);
  std::uniform_int_distribution<int> dist_rank(0, size - 1);

  ParticleSet initial_distribution(N, A->get_particle_spec());

  // determine which particles should end up on which rank
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = (px * 3) + dimx;
    }

    const auto px_cell = dist_cell(rng_cell);
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = px_cell;
    initial_distribution[Sym<INT>("ID")][px][0] = px;

    const auto px_rank = dist_rank(rng_rank);
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
  }

  A->add_particles_local(initial_distribution);

  BufferShared<INT> new_cell_ids(sycl_target, N);

  for (int testx = 0; testx < Ntest; testx++) {
    // set the new cells
    for (int px = 0; px < N; px++) {
      new_cell_ids.ptr[px] = dist_cell(rng_cell);
    }
    auto k_new_cell_ids = new_cell_ids.ptr;

    particle_loop(
        A,
        [=](auto index, auto ID) {
          ID.at(0) = k_new_cell_ids[index.get_loop_linear_index()];
        },
        Access::read(ParticleLoopIndex{}), Access::write(A->cell_id_dat))
        ->execute();

    A->reset_version_tracker();
    A->cell_move();
    A->test_version_different();
    A->test_internal_state();

    int npart_found = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto cell_id = A->get_cell(Sym<INT>("CELL_ID"), cellx);
      auto id = A->get_cell(Sym<INT>("ID"), cellx);
      auto v = A->get_cell(Sym<REAL>("V"), cellx);

      for (int rowx = 0; rowx < v->nrow; rowx++) {

        ASSERT_EQ((*cell_id)[0][rowx], cellx);

        const INT px = (*id)[0][rowx];
        for (int dimx = 0; dimx < 3; dimx++) {
          ASSERT_TRUE(ABS((px * 3 + dimx) - (*v)[dimx][rowx]) < 1.0e-10);
        }

        npart_found++;
      }
    }
    ASSERT_EQ(npart_found, N);
  }

  mesh->free();
}

TEST(ParticleGroup, cell_move_compression) {

  const int ndim = 2;
  std::vector<int> dims(3);
  dims[0] = (ndim == 2) ? 8 : 5;
  dims[1] = (ndim == 2) ? 5 : 4;
  dims[2] = 3;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  const int stencil_width = 1;

  auto mesh =
      std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, cell_extent,
                                       subdivision_order, stencil_width);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  // create object to map local cells + stencil to ranks
  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = make_test_obj<ParticleGroup>(domain, particle_spec, sycl_target);

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);
  std::mt19937 rng_rank(18241);

  const int N = 1024;
  const int Ntest = 1;
  const REAL dt = 1.0;
  const REAL tol = 1.0e-10;
  const int cell_count = domain->mesh->get_cell_count();

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities = NESO::Particles::normal_distribution(
      N, 3, 0.0, dims[0] * cell_extent, rng_vel);

  std::uniform_int_distribution<int> uniform_dist(
      0, sycl_target->comm_pair.size_parent - 1);

  ParticleSet initial_distribution(N, A->get_particle_spec());

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
      initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
    const auto px_rank = uniform_dist(rng_rank);
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
  }

  if (sycl_target->comm_pair.rank_parent == 0) {
    A->add_particles_local(initial_distribution);
  }

  CartesianPeriodic pbc(sycl_target, mesh, A->position_dat);

  reset_mpi_ranks(A->mpi_rank_dat);

  const auto k_ndim = ndim;
  const auto k_dt = dt;
  auto advect_loop = particle_loop(
      A,
      [=](auto k_V, auto k_P) {
        for (int dimx = 0; dimx < k_ndim; dimx++) {
          k_P.at(dimx) += k_V.at(dimx) * k_dt;
        }
      },
      Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("P")));


  std::vector<int> h_correct_compress_cells_old;
  std::vector<int> h_correct_compress_layers_old;
  std::vector<int> h_correct_compress_layers_new;
  std::vector<int> h_correct_npart_cell_new(mesh->get_cell_count());

  std::vector<int> h_to_test_compress_cells_old;
  std::vector<int> h_to_test_compress_layers_old;
  std::vector<int> h_to_test_compress_layers_new;
  std::vector<int> h_to_test_npart_cell_new(mesh->get_cell_count());

  const std::size_t ncell_int_num_bytes = mesh->get_cell_count() * sizeof(int);

  std::map<int, std::set<int>> map_cell_layers_old_correct;
  std::map<int, std::set<int>> map_cell_layers_new_correct;
  std::map<int, std::set<int>> map_cell_layers_old_to_test;
  std::map<int, std::set<int>> map_cell_layers_new_to_test;

  int compress_npart_correct;
  int compress_npart_to_test;

  auto lambda_callback_sparse =
      [&](int compress_npart, INT *k_compress_cells_old,
          INT *k_compress_layers_old, INT *k_compress_layers_new,
          int *k_npart_cell_new) {
        compress_npart_correct = compress_npart;
        nprint_variable(compress_npart_correct);
        h_correct_compress_cells_old .resize(compress_npart);
        h_correct_compress_layers_old.resize(compress_npart);
        h_correct_compress_layers_new.resize(compress_npart);
        const std::size_t num_bytes = sizeof(int) * compress_npart;
        sycl_target->queue.memcpy(h_correct_compress_cells_old.data(), k_compress_cells_old, num_bytes).wait();
        sycl_target->queue.memcpy(h_correct_compress_layers_old.data(), k_compress_layers_old, num_bytes).wait();
        sycl_target->queue.memcpy(h_correct_compress_layers_new.data(), k_compress_layers_new, num_bytes).wait();
        sycl_target->queue.memcpy(h_correct_npart_cell_new.data(), k_npart_cell_new, ncell_int_num_bytes).wait();

        map_cell_layers_old_correct.clear();
        map_cell_layers_new_correct.clear();
        for(int ix=0 ; ix<compress_npart ; ix++){
          const auto cell = h_correct_compress_cells_old.at(ix);
          const auto layer_old = h_correct_compress_layers_old.at(ix);
          const auto layer_new = h_correct_compress_layers_new.at(ix);
          map_cell_layers_old_correct[cell].insert(layer_old);
          map_cell_layers_new_correct[cell].insert(layer_new);
          if (cell == 0){
            nprint("SPARSE:", layer_old, layer_new);
          }
        }

      };

  auto lambda_callback_dense =
      [&](int compress_npart, INT *k_compress_cells_old,
          INT *k_compress_layers_old, INT *k_compress_layers_new,
          int *k_npart_cell_new) {

        compress_npart_to_test = compress_npart;
        nprint_variable(compress_npart_to_test);
        h_to_test_compress_cells_old .resize(compress_npart);
        h_to_test_compress_layers_old.resize(compress_npart);
        h_to_test_compress_layers_new.resize(compress_npart);
        const std::size_t num_bytes = sizeof(int) * compress_npart;
        sycl_target->queue.memcpy(h_to_test_compress_cells_old.data(), k_compress_cells_old, num_bytes).wait();
        sycl_target->queue.memcpy(h_to_test_compress_layers_old.data(), k_compress_layers_old, num_bytes).wait();
        sycl_target->queue.memcpy(h_to_test_compress_layers_new.data(), k_compress_layers_new, num_bytes).wait();
        sycl_target->queue.memcpy(h_to_test_npart_cell_new.data(), k_npart_cell_new, ncell_int_num_bytes).wait();

        map_cell_layers_old_to_test.clear();
        map_cell_layers_new_to_test.clear();
        for(int ix=0 ; ix<compress_npart ; ix++){
          const auto cell = h_to_test_compress_cells_old.at(ix);
          const auto layer_old = h_to_test_compress_layers_old.at(ix);
          const auto layer_new = h_to_test_compress_layers_new.at(ix);
          map_cell_layers_old_to_test[cell].insert(layer_old);
          map_cell_layers_new_to_test[cell].insert(layer_new);
          if (cell == 0){
            nprint("DENSE:", layer_old, layer_new);
          }
        }
      };

  auto lambda_callback_test = [&](){
        nprint("TEST CALLNACL");

        ASSERT_EQ(compress_npart_to_test, compress_npart_correct);
        ASSERT_EQ(h_correct_npart_cell_new, h_to_test_npart_cell_new);

        ASSERT_EQ(map_cell_layers_old_correct.size(), map_cell_layers_new_correct.size());
        for(int cx=0 ; cx<cell_count ; cx++){
          ASSERT_EQ(map_cell_layers_old_to_test[cx].size(), map_cell_layers_new_to_test[cx].size());
          //ASSERT_EQ(map_cell_layers_old_correct[cx].size(), map_cell_layers_new_correct[cx].size());
          //ASSERT_EQ(map_cell_layers_old_correct[cx].size(), map_cell_layers_old_to_test[cx].size());
          //ASSERT_EQ(map_cell_layers_new_correct[cx].size(), map_cell_layers_new_to_test[cx].size());
        }

  };

  nprint("========================================================================");
  nprint("========================================================================");
  nprint("========================================================================");
  nprint("========================================================================");
  A->cell_move_ctx.layer_compressor.test_mode = true;
  A->cell_move_ctx.layer_compressor.callback_sparse = lambda_callback_sparse;
  A->cell_move_ctx.layer_compressor.callback_dense = lambda_callback_dense;
  A->cell_move_ctx.layer_compressor.callback_test = lambda_callback_test;

  auto lambda_test = [&] {

  };

  for (int testx = 0; testx < Ntest; testx++) {
    pbc.execute();

    if (testx % 20 == 0) {
      A->reset_version_tracker();
    }
    A->hybrid_move();
    if (testx % 20 == 0) {
      A->test_version_different();
      A->test_internal_state();
    }

    A->cell_move();

    lambda_test();
    advect_loop->execute();
  }
  mesh->free();
}
