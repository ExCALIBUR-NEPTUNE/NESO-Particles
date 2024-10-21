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

TEST(ParticleLoop, local_memory) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  LocalMemory local_mem_real(7 * sizeof(REAL));
  auto local_mem_int = std::make_shared<LocalMemory>(3 * sizeof(INT));

  particle_loop(
      A,
      [=](auto INDEX, auto ID, auto OUT_REAL, auto OUT_INT, auto LM_REAL,
          auto LM_INT) {
        const auto index = INDEX.get_local_linear_index();
        REAL *ptr_real = static_cast<REAL *>(LM_REAL.data());
        INT *ptr_int = static_cast<INT *>(LM_INT.data());
        {
          for (int cx = 0; cx < 7; cx++) {
            ptr_real[cx] = index * 7 + cx;
          }
          for (int cx = 0; cx < 3; cx++) {
            ptr_int[cx] = index * 3 + cx;
          }
        }
        ID.at(0) = index;
        {
          for (int cx = 0; cx < 7; cx++) {
            OUT_REAL.at(cx) = ptr_real[cx];
          }
          for (int cx = 0; cx < 3; cx++) {
            OUT_INT.at(cx) = ptr_int[cx];
          }
        }
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("ID")),
      Access::write(Sym<REAL>("OUT_REAL")), Access::write(Sym<INT>("OUT_INT")),
      Access::write(local_mem_real), Access::write(local_mem_int))
      ->execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto out_real = A->get_dat(Sym<REAL>("OUT_REAL"))->cell_dat.get_cell(cellx);
    auto out_int = A->get_dat(Sym<INT>("OUT_INT"))->cell_dat.get_cell(cellx);
    auto index = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = out_real->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      auto idx = index->at(rowx, 0);
      for (int cx = 0; cx < 7; cx++) {
        ASSERT_NEAR(out_real->at(rowx, cx), idx * 7 + cx, 1.0e-15);
      }
      for (int cx = 0; cx < 3; cx++) {
        ASSERT_NEAR(out_int->at(rowx, cx), idx * 3 + cx, 1.0e-15);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
