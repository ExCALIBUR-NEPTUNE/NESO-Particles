#include "include/test_neso_particles.hpp"

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common(const int N = 1093) {
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;

  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 2),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int rank = sycl_target->comm_pair.rank_parent;
  const INT id_offset = rank * N;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px + id_offset;
  }

  A->add_particles_local(initial_distribution);
  parallel_advection_initialisation(A, 16);

  auto ccb = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, A->position_dat, A->cell_id_dat);

  ccb->execute();
  A->cell_move();

  return A;
}

} // namespace

TEST(ParticleLoop, sym_vector_pointer_cache) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  SymVectorPointerCache cache_real(sycl_target, &A->particle_dats_real);
  SymVectorPointerCache cache_int(sycl_target, &A->particle_dats_int);

  {
    auto lambda_test_const_pointers = [&](auto syms, auto &cache) {
      cache.create(syms);
      auto to_test_ptrs = cache.get_const(syms);
      ErrorPropagate ep(sycl_target);
      auto k_ep = ep.device_ptr();
      const std::size_t n = syms.size();

      particle_loop(
          A,
          [=](auto sym_vector) {
            for (std::size_t ix = 0; ix < n; ix++) {
              NESO_KERNEL_ASSERT(sym_vector.ptr[ix] == to_test_ptrs[ix], k_ep);
            }
          },
          Access::read(sym_vector(A, syms)))
          ->execute(0);

      EXPECT_FALSE(ep.get_flag());
    };

    auto lambda_test_pointers = [&](auto syms, auto &cache) {
      cache.create(syms);
      auto to_test_ptrs = cache.get(syms);
      ErrorPropagate ep(sycl_target);
      auto k_ep = ep.device_ptr();
      const std::size_t n = syms.size();

      particle_loop(
          A,
          [=](auto sym_vector) {
            for (std::size_t ix = 0; ix < n; ix++) {
              NESO_KERNEL_ASSERT(sym_vector.ptr[ix] == to_test_ptrs[ix], k_ep);
            }
          },
          Access::write(sym_vector(A, syms)))
          ->execute(0);

      EXPECT_FALSE(ep.get_flag());
    };

    {
      auto syms = std::vector({Sym<INT>("CELL_ID"), Sym<INT>("ID")});
      EXPECT_FALSE(cache_int.in_cache(syms));
      EXPECT_FALSE(cache_int.in_const_cache(syms));
      lambda_test_pointers(syms, cache_int);
      EXPECT_TRUE(cache_int.in_cache(syms));
      EXPECT_TRUE(cache_int.in_const_cache(syms));
    }
    {
      auto syms = std::vector({Sym<INT>("ID"), Sym<INT>("CELL_ID")});
      EXPECT_FALSE(cache_int.in_cache(syms));
      EXPECT_FALSE(cache_int.in_const_cache(syms));
      lambda_test_pointers(syms, cache_int);
      EXPECT_TRUE(cache_int.in_cache(syms));
      EXPECT_TRUE(cache_int.in_const_cache(syms));
    }
    cache_int.reset();
    {
      auto syms = std::vector({Sym<INT>("ID"), Sym<INT>("CELL_ID")});
      EXPECT_FALSE(cache_int.in_cache(syms));
      EXPECT_FALSE(cache_int.in_const_cache(syms));
      lambda_test_pointers(syms, cache_int);
      EXPECT_TRUE(cache_int.in_cache(syms));
      EXPECT_TRUE(cache_int.in_const_cache(syms));
    }
    {
      auto syms =
          std::vector({Sym<REAL>("P"), Sym<REAL>("V"), Sym<REAL>("P2")});
      EXPECT_FALSE(cache_real.in_cache(syms));
      EXPECT_FALSE(cache_real.in_const_cache(syms));
      lambda_test_pointers(syms, cache_real);
      EXPECT_TRUE(cache_real.in_cache(syms));
      EXPECT_TRUE(cache_real.in_const_cache(syms));
    }
    {
      auto syms =
          std::vector({Sym<REAL>("P"), Sym<REAL>("V"), Sym<REAL>("P2")});
      EXPECT_TRUE(cache_real.in_cache(syms));
      EXPECT_TRUE(cache_real.in_const_cache(syms));
      lambda_test_const_pointers(syms, cache_real);
      EXPECT_TRUE(cache_real.in_cache(syms));
      EXPECT_TRUE(cache_real.in_const_cache(syms));
    }
  }

  SymVectorPointerCacheDispatch dispatcher(sycl_target, &A->particle_dats_int,
                                           &A->particle_dats_real);

  {
    auto lambda_test_const_pointers = [&](auto syms) {
      dispatcher.create(syms);
      auto to_test_ptrs = dispatcher.get_const(syms);
      ErrorPropagate ep(sycl_target);
      auto k_ep = ep.device_ptr();
      const std::size_t n = syms.size();

      particle_loop(
          A,
          [=](auto sym_vector) {
            for (std::size_t ix = 0; ix < n; ix++) {
              NESO_KERNEL_ASSERT(sym_vector.ptr[ix] == to_test_ptrs[ix], k_ep);
            }
          },
          Access::read(sym_vector(A, syms)))
          ->execute(0);

      EXPECT_FALSE(ep.get_flag());
    };

    auto lambda_test_pointers = [&](auto syms) {
      dispatcher.create(syms);
      auto to_test_ptrs = dispatcher.get(syms);
      ErrorPropagate ep(sycl_target);
      auto k_ep = ep.device_ptr();
      const std::size_t n = syms.size();

      particle_loop(
          A,
          [=](auto sym_vector) {
            for (std::size_t ix = 0; ix < n; ix++) {
              NESO_KERNEL_ASSERT(sym_vector.ptr[ix] == to_test_ptrs[ix], k_ep);
            }
          },
          Access::write(sym_vector(A, syms)))
          ->execute(0);

      EXPECT_FALSE(ep.get_flag());
    };

    {
      auto syms = std::vector({Sym<INT>("CELL_ID"), Sym<INT>("ID")});
      EXPECT_FALSE(dispatcher.cache_int.in_cache(syms));
      EXPECT_FALSE(dispatcher.cache_int.in_const_cache(syms));
      lambda_test_pointers(syms);
      EXPECT_TRUE(dispatcher.cache_int.in_cache(syms));
      EXPECT_TRUE(dispatcher.cache_int.in_const_cache(syms));
    }
    {
      auto syms = std::vector({Sym<INT>("ID"), Sym<INT>("CELL_ID")});
      EXPECT_FALSE(dispatcher.cache_int.in_cache(syms));
      EXPECT_FALSE(dispatcher.cache_int.in_const_cache(syms));
      lambda_test_pointers(syms);
      EXPECT_TRUE(dispatcher.cache_int.in_cache(syms));
      EXPECT_TRUE(dispatcher.cache_int.in_const_cache(syms));
    }
    dispatcher.cache_int.reset();
    {
      auto syms = std::vector({Sym<INT>("ID"), Sym<INT>("CELL_ID")});
      EXPECT_FALSE(dispatcher.cache_int.in_cache(syms));
      EXPECT_FALSE(dispatcher.cache_int.in_const_cache(syms));
      lambda_test_pointers(syms);
      EXPECT_TRUE(dispatcher.cache_int.in_cache(syms));
      EXPECT_TRUE(dispatcher.cache_int.in_const_cache(syms));
    }
    {
      auto syms =
          std::vector({Sym<REAL>("P"), Sym<REAL>("V"), Sym<REAL>("P2")});
      EXPECT_FALSE(dispatcher.cache_real.in_cache(syms));
      EXPECT_FALSE(dispatcher.cache_real.in_const_cache(syms));
      lambda_test_pointers(syms);
      EXPECT_TRUE(dispatcher.cache_real.in_cache(syms));
      EXPECT_TRUE(dispatcher.cache_real.in_const_cache(syms));
    }
    {
      auto syms =
          std::vector({Sym<REAL>("P"), Sym<REAL>("V"), Sym<REAL>("P2")});
      EXPECT_TRUE(dispatcher.cache_real.in_cache(syms));
      EXPECT_TRUE(dispatcher.cache_real.in_const_cache(syms));
      lambda_test_const_pointers(syms);
      EXPECT_TRUE(dispatcher.cache_real.in_cache(syms));
      EXPECT_TRUE(dispatcher.cache_real.in_const_cache(syms));
    }
  }

  A->free();
  mesh->free();
}

TEST(ParticleLoop, sym_vector) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();

  auto aa = particle_sub_group(A);

  auto si = sym_vector(aa, {Sym<INT>("ID"), Sym<INT>("CELL_ID")});
  std::vector<Sym<REAL>> srv = {Sym<REAL>("V"), Sym<REAL>("P2")};
  auto sr = sym_vector(aa, srv);

  auto pl = particle_loop(
      aa,
      [=](auto index, auto dats_real, auto dats_int) {
        const INT cell = index.cell;
        const INT layer = index.layer;
        dats_real.at(1, cell, layer, 0) = dats_real.at(0, cell, layer, 0);
        dats_real.at(1, index, 1) = dats_int.at(0, 0);
      },
      Access::read(ParticleLoopIndex{}), Access::write(sr), Access::read(si));

  pl->execute();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto v = A->get_dat(Sym<REAL>("V"))->cell_dat.get_cell(cellx);
    auto p2 = A->get_dat(Sym<REAL>("P2"))->cell_dat.get_cell(cellx);
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = p2->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      // for each dimension
      ASSERT_EQ((*p2)[0][rowx], (*v)[0][rowx]);
      ASSERT_TRUE(std::abs((REAL)(*p2)[1][rowx] - (REAL)(*id)[0][rowx]) <
                  1.0e-12);
    }
  }

  A->free();
  mesh->free();
}
