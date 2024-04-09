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
};

template <> struct TestMapperToNP<TestParticleGroup> {
  using type = ParticleGroup;
};
template <> struct NPToTestMapper<ParticleGroup> {
  using type = TestParticleGroup;
};

} // namespace NESO::Particles

#endif
