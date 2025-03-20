#ifndef _NESO_PARTICLES_TEST_PARTICLE_GROUP_H_
#define _NESO_PARTICLES_TEST_PARTICLE_GROUP_H_

#include "test_base.hpp"

namespace NESO::Particles {

struct TestParticleGroup : public ParticleGroup {

  int test_ncell;
  ParticleDatVersionTracker t_particle_dat_versions;

  template <typename... ARGS>
  TestParticleGroup(ARGS... args) : ParticleGroup(args...) {
    test_ncell = ncell;
  }

  template <typename T> inline void check_dat(ParticleDatSharedPtr<T> dat) {
    std::vector<int> th_npart_cell_dat(ncell);
    std::vector<int> td_npart_cell_dat(ncell);
    this->sycl_target->queue
        .memcpy(th_npart_cell_dat.data(), dat->h_npart_cell,
                ncell * sizeof(int))
        .wait_and_throw();
    this->sycl_target->queue
        .memcpy(td_npart_cell_dat.data(), dat->d_npart_cell,
                ncell * sizeof(int))
        .wait_and_throw();
    auto th_npart_cell = this->h_npart_cell.get();
    for (int cx = 0; cx < ncell; cx++) {
      ASSERT_EQ(th_npart_cell_dat.at(cx), th_npart_cell.at(cx));
      ASSERT_EQ(td_npart_cell_dat.at(cx), th_npart_cell.at(cx));
      const int nrow_cell = dat->cell_dat.nrow_alloc.at(cx);
      ASSERT_TRUE(nrow_cell >= th_npart_cell.at(cx));
      ASSERT_TRUE(ncell == dat->cell_dat.ncells);
      ASSERT_TRUE(dat->cell_dat.get_nrow_max() >= th_npart_cell.at(cx));
      ASSERT_TRUE(dat->cell_dat.get_nrow_min() <= th_npart_cell.at(cx));
    }
    ASSERT_EQ(dat->ncell, ncell);
  }

  inline void test_init() {
    ASSERT_EQ(test_ncell, ncell);
    auto th_npart_cell = this->h_npart_cell.get();
    auto td_npart_cell = this->d_npart_cell.get();
    ASSERT_EQ(th_npart_cell.size(), ncell);
    ASSERT_EQ(td_npart_cell.size(), ncell);

    for (int cx = 0; cx < ncell; cx++) {
      ASSERT_EQ(th_npart_cell.at(cx), 0);
      ASSERT_EQ(td_npart_cell.at(cx), 0);
      ASSERT_EQ(this->get_npart_cell(cx), 0);
    }
    const auto npart_local = this->get_npart_local();
    ASSERT_EQ(0, npart_local);

    auto th_npart_cell_es = this->dh_npart_cell_es->h_buffer.get();
    auto td_npart_cell_es = this->dh_npart_cell_es->d_buffer.get();

    for (int cx = 0; cx < ncell; cx++) {
      ASSERT_EQ(th_npart_cell_es.at(cx), 0);
      ASSERT_EQ(td_npart_cell_es.at(cx), 0);
    }

    for (auto dat : this->particle_dats_real) {
      this->check_dat(dat.second);
    }
    for (auto dat : this->particle_dats_real) {
      this->check_dat(dat.second);
    }
  }

  inline void test_internal_state() {
    ASSERT_EQ(test_ncell, ncell);
    auto th_npart_cell = this->h_npart_cell.get();
    auto td_npart_cell = this->d_npart_cell.get();

    ASSERT_EQ(th_npart_cell.size(), ncell);
    ASSERT_EQ(td_npart_cell.size(), ncell);

    INT t_npart_local = 0;
    for (int cx = 0; cx < ncell; cx++) {
      ASSERT_EQ(th_npart_cell.at(cx), td_npart_cell.at(cx));
      t_npart_local += th_npart_cell.at(cx);
      ASSERT_EQ(this->get_npart_cell(cx), td_npart_cell.at(cx));
    }
    const auto npart_local = this->get_npart_local();
    ASSERT_EQ(t_npart_local, npart_local);

    auto th_npart_cell_es = this->dh_npart_cell_es->h_buffer.get();
    auto td_npart_cell_es = this->dh_npart_cell_es->d_buffer.get();

    ASSERT_EQ(th_npart_cell_es.size(), ncell);
    ASSERT_EQ(td_npart_cell_es.size(), ncell);

    std::vector<INT> npart_cell_es(ncell);
    std::exclusive_scan(th_npart_cell.begin(), th_npart_cell.end(),
                        npart_cell_es.begin(), 0);

    for (int cx = 0; cx < ncell; cx++) {
      ASSERT_EQ(th_npart_cell_es.at(cx), npart_cell_es.at(cx));
      ASSERT_EQ(td_npart_cell_es.at(cx), npart_cell_es.at(cx));
    }

    for (auto dat : this->particle_dats_real) {
      this->check_dat(dat.second);
    }
    for (auto dat : this->particle_dats_real) {
      this->check_dat(dat.second);
    }
  }

  inline void reset_version_tracker() {
    auto lambda_dispatch = [&](auto dat) {
      this->t_particle_dat_versions[dat->sym] = 0;
    };
    for (auto dat : this->particle_dats_real) {
      lambda_dispatch(dat.second);
    }
    for (auto dat : this->particle_dats_real) {
      lambda_dispatch(dat.second);
    }
    this->check_validation(this->t_particle_dat_versions);
  }

  inline void test_version_different() {
    ASSERT_TRUE(this->check_validation(this->t_particle_dat_versions));
  }

  inline void set_local_move_context(std::vector<int> &ranks) {
    this->local_move_ctx = std::make_unique<LocalMove>(
        sycl_target, layer_compressor, particle_dats_real, particle_dats_int,
        ranks.size(), ranks.data());
    this->local_move_ctx->set_mpi_rank_dat(mpi_rank_dat);
  }

  inline auto get_particle_group_version() {
    return this->particle_group_version;
  }
  inline bool wrap_check_validation(int64_t &to_check) {
    return this->check_validation(to_check);
  }
};

template <> struct TestMapperToNP<TestParticleGroup> {
  using type = ParticleGroup;
};
template <> struct NPToTestMapper<ParticleGroup> {
  using type = TestParticleGroup;
};

inline std::tuple<ParticleGroupSharedPtr, SYCLTargetSharedPtr, int>
particle_loop_common_2d(const int npart_cell = 1093, const int nx = 16,
                        const int ny = 32) {
  constexpr int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = nx;
  dims[1] = ny;

  const double cell_extent = 1.0;
  const int subdivision_order = 0;
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

  const int cell_count = mesh->get_cell_count();
  const int N = cell_count * npart_cell;
  int id_offset;
  MPICHK(MPI_Exscan(&N, &id_offset, 1, MPI_INT, MPI_SUM, mesh->get_comm()));

  const int rank = sycl_target->comm_pair.rank_parent;

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

  return {A, sycl_target, cell_count};
}

} // namespace NESO::Particles

#endif
