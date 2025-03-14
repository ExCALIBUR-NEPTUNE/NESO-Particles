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

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) == 0; },
      Access::read(Sym<INT>("CELL_ID")));
  A->remove_particles(aa);

  return A;
}

} // namespace

class ParticleLoopLocalMem : public testing::TestWithParam<std::size_t> {};
TEST_P(ParticleLoopLocalMem, iteration_set_base) {
  auto sycl_target = std::make_shared<SYCLTarget>(GPU_SELECTOR, MPI_COMM_WORLD);
  const std::size_t Nbin = 16;
  const std::size_t Ncell = 197;
  const std::size_t local_size = 32;
  const std::size_t Nmin = 65;
  const std::size_t Nmax = 1023;

  // Some of the SYCL implementations claim to have an extremely large local
  // memory limit by default that we cannot reasonably test, e.g.
  // 18446744073709551615 bytes.
  std::size_t local_mem_required = GetParam();
  std::size_t default_limit = 8 * local_mem_required + 3;
  std::size_t existing_limit = sycl_target->device_limits.local_mem_size;
  if (existing_limit < default_limit) {
    local_mem_required = existing_limit / 2;
  } else {
    sycl_target->device_limits.local_mem_size = default_limit;
  }
  const std::size_t local_mem_limit = sycl_target->device_limits.local_mem_size;

  std::uniform_int_distribution<std::size_t> dst(Nmin, Nmax);
  std::mt19937 rng(5223423);
  std::vector<int> h_npart_cell(Ncell);
  std::generate(h_npart_cell.begin(), h_npart_cell.end(),
                [&]() { return dst(rng); });

  ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{
      sycl_target, Ncell, h_npart_cell.data(), h_npart_cell.data()};

  // Test the single cell iteration set
  {
    for (std::size_t cellx = 0; cellx < Ncell; cellx++) {
      auto is = ish.get_single_cell(cellx, local_size, local_mem_required);
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

        if (local_mem_required == 0) {
          ASSERT_EQ(range_local.get(1), local_size);
        } else {
          ASSERT_TRUE(range_local.get(1) * local_mem_required <=
                      local_mem_limit);
        }

        ASSERT_EQ(range_global.get(1) % range_local.get(1), 0);
        offset_layer += range_global.get(1);

        if (static_cast<std::size_t>(h_npart_cell.at(cellx)) < local_size) {
          ASSERT_EQ(nblocks, 1);
          EXPECT_TRUE(blockx.layer_bounds_check_required);
        } else if (static_cast<std::size_t>(h_npart_cell.at(cellx)) %
                       local_size ==
                   0) {
          ASSERT_FALSE(blockx.layer_bounds_check_required);
          ASSERT_EQ(nblocks, 1);
        } else {
          if (ix == 0) {
            ASSERT_FALSE(blockx.layer_bounds_check_required);
          } else {
            ASSERT_TRUE(blockx.layer_bounds_check_required);
          }
        }
        if (ix == (nblocks - 1)) {
          EXPECT_TRUE(offset_layer >=
                      static_cast<std::size_t>(h_npart_cell.at(cellx)));
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
      for (int ix = 0; ix < h_npart_cell.at(cellx); ix++) {
        set_correct.insert(static_cast<std::size_t>(ix));
      }

      EXPECT_EQ(set_correct.size(), set_test.size());
      EXPECT_EQ(set_correct, set_test);
    }
  }

  // Test the all cell iteration set
  {
    std::set<std::array<std::size_t, 2>> set_test;
    auto is = ish.get_all_cells(Nbin, local_size, local_mem_required);

    for (auto &blockx : is) {
      auto range_global = blockx.loop_iteration_set.get_global_range();
      auto range_local = blockx.loop_iteration_set.get_local_range();
      EXPECT_EQ(range_local.get(0), 1);

      if (local_mem_required == 0) {
        EXPECT_EQ(range_local.get(1), local_size);
      } else {
        ASSERT_TRUE(range_local.get(1) * local_mem_required <= local_mem_limit);
      }
      ASSERT_EQ(range_global.get(1) % range_local.get(1), 0);

      const std::size_t cell_start = blockx.block_device.offset_cell;
      const std::size_t cell_end = range_global.get(0) + cell_start;
      const std::size_t layer_start = blockx.block_device.offset_layer;
      const std::size_t layer_end = range_global.get(1) + layer_start;
      for (std::size_t cell = cell_start; cell < cell_end; cell++) {
        for (std::size_t layer = layer_start; layer < layer_end; layer++) {
          if (blockx.layer_bounds_check_required) {
            if (blockx.block_device.work_item_required(cell, layer)) {
              set_test.insert({cell, layer});
            }
          } else {
            set_test.insert({cell, layer});
          }
        }
      }
    }

    std::set<std::array<std::size_t, 2>> set_correct;
    for (std::size_t cell = 0; cell < Ncell; cell++) {
      const auto npart_cell = static_cast<std::size_t>(h_npart_cell.at(cell));
      for (std::size_t layer = 0; layer < npart_cell; layer++) {
        set_correct.insert({cell, layer});
      }
    }
    EXPECT_EQ(set_correct.size(), set_test.size());
    EXPECT_EQ(set_correct, set_test);
  }

  sycl_target->free();
}

TEST(ParticleLoop, iteration_set) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  particle_loop(
      A,
      [=](auto INDEX, auto OUT_INT) {
        OUT_INT.at(0) = INDEX.get_loop_linear_index();
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("OUT_INT")))
      ->execute();

  ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{
      A->mpi_rank_dat};

  auto ptr = A->get_dat(Sym<INT>("OUT_INT"))->cell_dat.device_ptr();

  EventStack es;
  auto is = ish.get_all_cells();
  for (auto &blockx : is) {
    const auto block_device = blockx.block_device;
    es.push(sycl_target->queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
        std::size_t cell;
        std::size_t layer;
        block_device.get_cell_layer(idx, &cell, &layer);
        if (block_device.work_item_required(cell, layer)) {
          ptr[cell][1][layer] = ptr[cell][0][layer];
        }
      });
    }));
  }
  es.wait();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  particle_loop(
      A,
      [=](auto OUT_INT) {
        NESO_KERNEL_ASSERT(OUT_INT.at(0) == OUT_INT.at(1), k_ep);
      },
      Access::read(Sym<INT>("OUT_INT")))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST_P(ParticleLoopLocalMem, iteration_set_base_stride) {
  auto sycl_target = std::make_shared<SYCLTarget>(GPU_SELECTOR, MPI_COMM_WORLD);
  const std::size_t Nbin = 13;
  const std::size_t Ncell = 197;
  const std::size_t local_size = 32;
  const std::size_t Nmin = 65;
  const std::size_t Nmax = 1023;
  const std::size_t stride = 7;
  std::uniform_int_distribution<std::size_t> dst(Nmin, Nmax);
  std::mt19937 rng(5223423);
  std::vector<int> h_npart_cell(Ncell);
  std::generate(h_npart_cell.begin(), h_npart_cell.end(),
                [&]() { return dst(rng); });

  // Some of the SYCL implementations claim to have an extremely large local
  // memory limit by default that we cannot reasonably test, e.g.
  // 18446744073709551615 bytes.
  std::size_t local_mem_required = GetParam();
  std::size_t default_limit = 8 * local_mem_required + 3;
  std::size_t existing_limit = sycl_target->device_limits.local_mem_size;
  if (existing_limit < default_limit) {
    local_mem_required = existing_limit / 2;
  } else {
    sycl_target->device_limits.local_mem_size = default_limit;
  }
  const std::size_t local_mem_limit = sycl_target->device_limits.local_mem_size;

  ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{
      sycl_target, Ncell, h_npart_cell.data(), h_npart_cell.data()};

  // Test the single cell iteration set
  {
    for (std::size_t cellx = 0; cellx < Ncell; cellx++) {
      auto is =
          ish.get_single_cell(cellx, local_size, local_mem_required, stride);
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
        if (local_mem_required == 0) {
          EXPECT_EQ(range_local.get(1), local_size);
        } else {
          ASSERT_TRUE(range_local.get(1) * local_mem_required * stride <=
                      local_mem_limit);
        }
        const std::size_t local_size_actual = range_local.get(1);
        ASSERT_EQ(range_global.get(1) % range_local.get(1), 0);

        if (nblocks == 1) {
          ASSERT_TRUE(range_global.get(1) * stride >=
                      static_cast<std::size_t>(h_npart_cell.at(cellx)));
        }

        offset_layer += range_global.get(1);

        if (static_cast<std::size_t>(h_npart_cell.at(cellx)) <
            local_size_actual * stride) {
          ASSERT_EQ(nblocks, 1);
          EXPECT_TRUE(blockx.layer_bounds_check_required);
        } else if (static_cast<std::size_t>(h_npart_cell.at(cellx)) %
                       (local_size_actual * stride) ==
                   0) {
          ASSERT_FALSE(blockx.layer_bounds_check_required);
          ASSERT_EQ(nblocks, 1);
        } else {
          if (ix == 0) {
            ASSERT_FALSE(blockx.layer_bounds_check_required);
          } else {
            ASSERT_TRUE(blockx.layer_bounds_check_required);
          }
        }

        EXPECT_EQ(blockx.block_device.d_npart_cell[cellx],
                  h_npart_cell.at(cellx));

        for (std::size_t iy = 0; iy < range_global.get(1); iy++) {
          std::size_t layer_block = iy + blockx.block_device.offset_layer;
          if (blockx.block_device.stride_work_item_required(cellx,
                                                            layer_block)) {
            const auto bound = blockx.block_device.stride_local_index_bound(
                cellx, layer_block);
            for (std::size_t lx = 0; lx < bound; lx++) {
              set_test.insert(layer_block * stride + lx);
            }
          }
        }
      }

      ASSERT_TRUE(offset_layer * stride >=
                  static_cast<std::size_t>(h_npart_cell.at(cellx)));

      std::set<std::size_t> set_correct;
      for (int ix = 0; ix < h_npart_cell.at(cellx); ix++) {
        set_correct.insert(static_cast<std::size_t>(ix));
      }

      EXPECT_EQ(set_correct.size(), set_test.size());
      EXPECT_EQ(set_correct, set_test);
    }
  }
  // test all cell iteration set
  {
    std::set<std::array<std::size_t, 2>> set_test;
    auto is = ish.get_all_cells(Nbin, local_size, local_mem_required, stride);

    for (auto &blockx : is) {
      auto range_global = blockx.loop_iteration_set.get_global_range();
      auto range_local = blockx.loop_iteration_set.get_local_range();
      EXPECT_EQ(range_local.get(0), 1);

      if (local_mem_required == 0) {
        EXPECT_EQ(range_local.get(1), local_size);
      } else {
        ASSERT_TRUE(range_local.get(1) * local_mem_required * stride <=
                    local_mem_limit);
      }
      ASSERT_EQ(range_global.get(1) % range_local.get(1), 0);

      const std::size_t cell_start = blockx.block_device.offset_cell;
      const std::size_t cell_end = range_global.get(0) + cell_start;
      const std::size_t block_start = blockx.block_device.offset_layer;
      const std::size_t block_end = range_global.get(1) + block_start;
      for (std::size_t cell = cell_start; cell < cell_end; cell++) {
        for (std::size_t block = block_start; block < block_end; block++) {
          if (blockx.block_device.stride_work_item_required(cell, block)) {
            const auto bound =
                blockx.block_device.stride_local_index_bound(cell, block);
            const std::size_t layer_start = block * stride;
            const std::size_t layer_end = layer_start + bound;
            for (std::size_t layer = layer_start; layer < layer_end; layer++) {
              set_test.insert({cell, layer});
            }
          }
        }
      }
    }

    std::set<std::array<std::size_t, 2>> set_correct;
    for (std::size_t cell = 0; cell < Ncell; cell++) {
      const auto npart_cell = static_cast<std::size_t>(h_npart_cell.at(cell));
      for (std::size_t layer = 0; layer < npart_cell; layer++) {
        set_correct.insert({cell, layer});
      }
    }
    EXPECT_EQ(set_correct.size(), set_test.size());
    EXPECT_EQ(set_correct, set_test);
  }

  sycl_target->free();
}

TEST(ParticleLoop, iteration_set_stride_single_cell) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const std::size_t stride = 3;
  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  particle_loop(
      A,
      [=](auto INDEX, auto OUT_INT) {
        OUT_INT.at(0) = INDEX.get_loop_linear_index();
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("OUT_INT")))
      ->execute();

  ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{
      A->mpi_rank_dat};

  auto ptr = A->get_dat(Sym<INT>("OUT_INT"))->cell_dat.device_ptr();

  EventStack es;
  const std::size_t cell_count = mesh->get_cell_count();
  for (std::size_t cellx = 0; cellx < cell_count; cellx++) {
    auto is = ish.get_single_cell(cellx, local_size, 0, stride);
    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      es.push(sycl_target->queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(
            blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
              std::size_t cell;
              std::size_t block;
              block_device.stride_get_cell_block(idx, &cell, &block);
              if (block_device.stride_work_item_required(cell, block)) {
                const std::size_t layer_start = block * stride;
                const std::size_t layer_end =
                    layer_start +
                    block_device.stride_local_index_bound(cell, block);
                for (std::size_t layer = layer_start; layer < layer_end;
                     layer++) {
                  ptr[cell][1][layer] = ptr[cell][0][layer];
                }
              }
            });
      }));
    }
  }
  es.wait();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  particle_loop(
      A,
      [=](auto OUT_INT) {
        NESO_KERNEL_ASSERT(OUT_INT.at(0) == OUT_INT.at(1), k_ep);
      },
      Access::read(Sym<INT>("OUT_INT")))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoop, iteration_set_stride_all_cells) {
  const std::size_t Nbin = 16;
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const std::size_t stride = 3;
  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  particle_loop(
      A,
      [=](auto INDEX, auto OUT_INT) {
        OUT_INT.at(0) = INDEX.get_loop_linear_index();
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("OUT_INT")))
      ->execute();

  ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{
      A->mpi_rank_dat};

  auto ptr = A->get_dat(Sym<INT>("OUT_INT"))->cell_dat.device_ptr();

  EventStack es;
  auto is = ish.get_all_cells(Nbin, local_size, 0, stride);
  for (auto &blockx : is) {
    const auto block_device = blockx.block_device;
    es.push(sycl_target->queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
        std::size_t cell;
        std::size_t block;
        block_device.stride_get_cell_block(idx, &cell, &block);
        if (block_device.stride_work_item_required(cell, block)) {
          const std::size_t layer_start = block * stride;
          const std::size_t layer_end =
              layer_start + block_device.stride_local_index_bound(cell, block);
          for (std::size_t layer = layer_start; layer < layer_end; layer++) {
            ptr[cell][1][layer] = ptr[cell][0][layer];
          }
        }
      });
    }));
  }
  es.wait();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  particle_loop(
      A,
      [=](auto OUT_INT) {
        NESO_KERNEL_ASSERT(OUT_INT.at(0) == OUT_INT.at(1), k_ep);
      },
      Access::read(Sym<INT>("OUT_INT")))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  A->free();
  sycl_target->free();
  mesh->free();
}

INSTANTIATE_TEST_SUITE_P(init, ParticleLoopLocalMem, testing::Values(0, 8));
