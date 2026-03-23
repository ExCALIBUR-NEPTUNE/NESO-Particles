#ifndef _NESO_PARTICLES_TEST_PARTICLE_GROUP_H_
#define _NESO_PARTICLES_TEST_PARTICLE_GROUP_H_

#include "test_base.hpp"

namespace NESO::Particles {

struct TestParticleGroup : public ParticleGroup {

  inline void set_zero_values(const REAL value_real, const INT value_int) {
    this->zero_value_real = value_real;
    this->zero_value_int = value_int;
  }

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
        sycl_target, layer_compressor, this->particle_group_pointer_map,
        ranks.size(), ranks.data());
    this->local_move_ctx->set_mpi_rank_dat(mpi_rank_dat);
  }

  inline auto get_particle_group_version() {
    return this->particle_group_version;
  }
  inline bool wrap_check_validation(std::int64_t &to_check) {
    return this->check_validation(to_check);
  }

  inline std::shared_ptr<ParticleGroupPointerMap>
  get_particle_group_pointer_map() {
    return this->particle_group_pointer_map;
  }

  inline void
  add_particles_local_old(std::shared_ptr<ParticleGroup> particle_group) {

    // Get the new cell occupancies
    const int cell_count = this->domain->mesh->get_cell_count();
    std::vector<INT> h_npart_cell_existing(cell_count);
    std::vector<INT> h_npart_cell_to_add(cell_count);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      h_npart_cell_existing.at(cellx) = this->h_npart_cell.ptr[cellx];
      const INT to_add = particle_group->h_npart_cell.ptr[cellx];
      this->h_npart_cell.ptr[cellx] += to_add;
      h_npart_cell_to_add.at(cellx) = to_add;
    }
    buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait_and_throw();
    this->recompute_npart_cell_es();

    // Allocate space in all the dats for the new particles
    this->realloc_all_dats();
    EventStack es;

    auto lambda_dispatch = [&](auto dat_dst) {
      // get a shared ptr of the right type
      auto dat_src = dat_dst;
      dat_src = nullptr;
      auto sym = dat_dst->sym;
      if (particle_group->contains_dat(sym)) {
        dat_src = particle_group->get_dat(sym);
      }
      dat_dst->append_particle_data(dat_src, h_npart_cell_existing,
                                    h_npart_cell_to_add, es);
    };

    // launch the copies of the npart cells into the ParticleDats
    for (auto &dat : this->particle_dats_real) {
      lambda_dispatch(dat.second);
    }
    for (auto &dat : this->particle_dats_int) {
      lambda_dispatch(dat.second);
    }

    // launch the copies of the npart cells into the ParticleDats
    for (auto &dat : this->particle_dats_real) {
      es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
    }
    for (auto &dat : this->particle_dats_int) {
      es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
    }
    es.wait();
    this->invalidate_group_version();
  }
};

template <> struct TestMapperToNP<TestParticleGroup> {
  using type = ParticleGroup;
};
template <> struct NPToTestMapper<ParticleGroup> {
  using type = TestParticleGroup;
};

inline std::tuple<ParticleGroupSharedPtr, SYCLTargetSharedPtr, int>
particle_loop_create_common(const int npart_cell = 10, const int ndim = 2,
                            const int nx = 16, const int ny = 32,
                            const int nz = 48, const REAL cell_extent = 1.0) {
  std::vector<int> dims(ndim);
  dims[0] = nx;
  dims[1] = ny;
  if (ndim > 2) {
    dims[2] = nz;
  }

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

  auto A =
      std::make_shared<TestParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int cell_count = mesh->get_cell_count();
  const int N = cell_count * npart_cell;
  int id_offset = 0;
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

inline std::tuple<ParticleGroupSharedPtr, SYCLTargetSharedPtr, int>
particle_loop_common_2d(const int npart_cell = 1093, const int nx = 16,
                        const int ny = 32, const REAL cell_extent = 1.0) {
  return particle_loop_create_common(npart_cell, 2, nx, ny, -1, cell_extent);
}

inline std::tuple<ParticleGroupSharedPtr, SYCLTargetSharedPtr, int>
particle_loop_common_3d(const int npart_cell = 1093, const int nx = 16,
                        const int ny = 32, const int nz = 48,
                        const REAL cell_extent = 1.0) {
  return particle_loop_create_common(npart_cell, 3, nx, ny, nz, cell_extent);
}

inline bool selection_is_self_consistent(
    ParticleSubGroupImplementation::Selection selection, const int cell_count,
    const int npart_local, SYCLTargetSharedPtr sycl_target,
    std::set<std::tuple<int, int>> &correct_pairs) {
  bool success = true;
  auto check = [&](const bool cond) -> void {
    NESOASSERT(cond, "uncomment to get throw in correct place");
    success = success && cond;
  };

  check(correct_pairs.size() == static_cast<std::size_t>(npart_local));
  check(npart_local == selection.npart_local);
  check(cell_count == selection.ncell);

  std::vector<int> h_npart_cell(cell_count);
  sycl_target->queue
      .memcpy(h_npart_cell.data(), selection.d_npart_cell,
              cell_count * sizeof(int))
      .wait_and_throw();
  std::vector<INT> h_npart_cell_es(cell_count);
  sycl_target->queue
      .memcpy(h_npart_cell_es.data(), selection.d_npart_cell_es,
              cell_count * sizeof(INT))
      .wait_and_throw();

  INT total = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    INT v = static_cast<INT>(selection.h_npart_cell[cx]);
    check(h_npart_cell.at(cx) == selection.h_npart_cell[cx]);
    check(h_npart_cell_es.at(cx) == total);
    total += v;
  }
  check(total == static_cast<INT>(npart_local));

  auto host_to_test_map =
      get_host_map_cells_to_particles(sycl_target, selection);

  std::set<std::tuple<int, int>> to_test_pairs;
  for (int cx = 0; cx < cell_count; cx++) {
    auto m = host_to_test_map.at(cx);
    check(static_cast<int>(m.size()) == h_npart_cell.at(cx));
    for (int rx : m) {
      std::tuple<int, int> key = {cx, rx};
      check(to_test_pairs.count(key) == 0);
      to_test_pairs.insert(key);
    }
  }

  check(to_test_pairs == correct_pairs);

  return success;
}

} // namespace NESO::Particles

#endif
