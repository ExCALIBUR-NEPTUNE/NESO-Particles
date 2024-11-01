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
                             ParticleProp(Sym<REAL>("OUT_REAL"), 7),
                             ParticleProp(Sym<INT>("OUT_INT"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
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

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
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

TEST(ParticleLoop, iteration_set_base) {
  auto sycl_target = std::make_shared<SYCLTarget>(GPU_SELECTOR, MPI_COMM_WORLD);
  const std::size_t Nbin = 16;
  const std::size_t Ncell = 197;
  const std::size_t local_size = 32;
  const std::size_t Nmin = 65;
  const std::size_t Nmax = 1023;
  std::uniform_int_distribution<std::size_t> dst(Nmin, Nmax);
  std::mt19937 rng(5223423);
  std::vector<int> h_npart_cell(Ncell);
  std::generate(h_npart_cell.begin(), h_npart_cell.end(),
                [&]() { return dst(rng); });

  ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{
      sycl_target, Nbin, Ncell, h_npart_cell.data(), h_npart_cell.data()};

  // Test the single cell iteration set
  {
    for (std::size_t cellx = 0; cellx < Ncell; cellx++) {
      auto is = ish.get_single_cell(cellx, local_size);
      const auto nblocks = is.size();

      std::size_t offset_layer = 0;
      std::set<std::size_t> set_test;

      for (std::size_t ix = 0; ix < nblocks; ix++) {
        auto blockx = is.at(ix);
        EXPECT_EQ(blockx.block_device.offset_cell, cellx);
        EXPECT_EQ(blockx.block_device.offset_layer, offset_layer);
        auto range_global = blockx.loop_iteration_set.get_global_range();
        auto range_local = blockx.loop_iteration_set.get_local_range();
        ASSERT_EQ(range_global.get(0), 1);
        ASSERT_EQ(range_local.get(0), 1);
        ASSERT_EQ(range_local.get(1), local_size);
        ASSERT_EQ(range_global.get(1) % local_size, 0);
        offset_layer += range_global.get(1);

        if (ix == (nblocks - 1)) {
          EXPECT_TRUE(blockx.layer_bounds_check_required);
          EXPECT_TRUE(offset_layer >= h_npart_cell.at(cellx));
        } else {
          EXPECT_FALSE(blockx.layer_bounds_check_required);
        }

        EXPECT_EQ(blockx.block_device.d_npart_cell[cellx],
                  h_npart_cell.at(cellx));

        for (std::size_t iy = 0; iy < range_global.get(1); iy++) {
          std::size_t layer = iy + blockx.block_device.offset_layer;
          if (blockx.block_device.work_item_required(cellx, layer)) {
            set_test.insert(layer);
          }
        }
      }

      std::set<std::size_t> set_correct;
      for (std::size_t ix = 0; ix < h_npart_cell.at(cellx); ix++) {
        set_correct.insert(ix);
      }

      EXPECT_EQ(set_correct.size(), set_test.size());
      EXPECT_EQ(set_correct, set_test);
    }
  }

  // Test the all cell iteration set

  sycl_target->free();
}

TEST(ParticleLoop, iteration_set) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  A->free();
  sycl_target->free();
  mesh->free();
}
