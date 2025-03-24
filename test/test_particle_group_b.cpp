#include "include/test_neso_particles.hpp"

TEST(ParticleGroup, temporary) {

  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);
  const int cell_count = cell_count_t;

  auto lambda_test_empty = [&](auto B) {
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(B->get_npart_cell(cx), 0);
    }
  };

  auto lambda_test = [&](auto A, auto B) {
    ASSERT_EQ(A->domain.get(), B->domain.get());
    ASSERT_EQ(A->sycl_target.get(), B->sycl_target.get());
    ASSERT_EQ(A->mpi_rank_dat->sym, B->mpi_rank_dat->sym);
    ASSERT_EQ(A->position_dat->sym, B->position_dat->sym);
    ASSERT_EQ(A->cell_id_dat->sym, B->cell_id_dat->sym);
    ASSERT_EQ(*A->cell_id_sym, *B->cell_id_sym);
    ASSERT_EQ(*A->position_sym, *B->position_sym);
    ASSERT_EQ(*A->mpi_rank_sym, *B->mpi_rank_sym);

    ASSERT_EQ(A->mpi_rank_dat.get(),
              A->get_dat(Sym<INT>("NESO_MPI_RANK")).get());
    ASSERT_EQ(B->mpi_rank_dat.get(),
              B->get_dat(Sym<INT>("NESO_MPI_RANK")).get());
    ASSERT_EQ(A->position_dat.get(), A->get_dat(Sym<REAL>("P")).get());
    ASSERT_EQ(B->position_dat.get(), B->get_dat(Sym<REAL>("P")).get());
    ASSERT_EQ(A->cell_id_dat.get(), A->get_dat(Sym<INT>("CELL_ID")).get());
    ASSERT_EQ(B->cell_id_dat.get(), B->get_dat(Sym<INT>("CELL_ID")).get());

    auto lambda_assert_dats = [&](auto a, auto b) {
      auto lambda_inner = [&](auto &datmap) {
        for (auto &[sym, dat] : datmap) {
          ASSERT_TRUE(b->contains_dat(sym));
          ASSERT_EQ(dat->ncomp, b->get_dat(sym)->ncomp);
        }
      };
      lambda_inner(a->particle_dats_real);
      lambda_inner(a->particle_dats_int);
    };
    lambda_assert_dats(A, B);
    lambda_assert_dats(B, A);

    B->add_particles_local(A);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(A->get_npart_cell(cx), B->get_npart_cell(cx));
    }
  };

  auto pgtmp = std::make_shared<ParticleGroupTemporary>();

  auto B = pgtmp->get(A);
  lambda_test_empty(B);
  lambda_test(A, B);
  auto C = pgtmp->get(A);
  lambda_test_empty(C);

  auto Bptr = B.get();
  auto Cptr = C.get();
  ASSERT_TRUE(Bptr != Cptr);

  lambda_test(A, C);
  pgtmp->restore(A, B);
  pgtmp->restore(A, C);

  B = pgtmp->get(A);

  ASSERT_TRUE(B.get() == Bptr || B.get() == Cptr);
  lambda_test_empty(B);
  lambda_test(A, B);
  pgtmp->restore(A, B);

  A->remove_particle_dat(Sym<REAL>("FOO"));
  B = pgtmp->get(A);
  lambda_test(A, B);
  pgtmp->restore(A, B);

  A->add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<INT>("BAR"), 4),
                                  A->domain->mesh->get_cell_count()));
  B = pgtmp->get(A);
  lambda_test(A, B);
  pgtmp->restore(A, B);

  sycl_target->free();
}

TEST(ParticleGroup, particle_group_pointer_map) {
  auto [A_t, sycl_target_t, cell_count_t] = particle_loop_common_2d(2, 16, 32);
  auto A = A_t;
  auto sycl_target = sycl_target_t;

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));
  ASSERT_TRUE(aa->create_if_required());
  ASSERT_FALSE(aa->create_if_required());

  aa->create_if_required();
  ASSERT_FALSE(aa->create_if_required());

  auto AT = std::static_pointer_cast<TestParticleGroup>(A);
  auto ptr_map = AT->get_particle_group_pointer_map();

  ASSERT_FALSE(aa->create_if_required());
  ptr_map->get_const();
  ASSERT_FALSE(aa->create_if_required());
  ptr_map->get();
  ASSERT_TRUE(aa->create_if_required());
  ASSERT_FALSE(aa->create_if_required());

  auto lambda_test = [&]() {
    auto AT = std::static_pointer_cast<TestParticleGroup>(A);
    auto ptr_map = AT->get_particle_group_pointer_map();

    auto d_ptr_const_map = ptr_map->get_const();
    auto d_ptr_map = ptr_map->get();

    int ncomp_total_real_correct = 0;
    for (auto &[sym, dat] : A->particle_dats_real) {
      ncomp_total_real_correct += dat->ncomp;
    }
    int ncomp_total_int_correct = 0;
    for (auto &[sym, dat] : A->particle_dats_int) {
      ncomp_total_int_correct += dat->ncomp;
    }

    ASSERT_EQ(d_ptr_const_map.d_ncomp_real, d_ptr_map.d_ncomp_real);
    ASSERT_EQ(d_ptr_const_map.d_ncomp_int, d_ptr_map.d_ncomp_int);
    ASSERT_EQ(d_ptr_const_map.d_ncomp_exscan_real,
              d_ptr_map.d_ncomp_exscan_real);
    ASSERT_EQ(d_ptr_const_map.d_ncomp_exscan_int, d_ptr_map.d_ncomp_exscan_int);
    ASSERT_EQ(d_ptr_const_map.h_ncomp_real, d_ptr_map.h_ncomp_real);
    ASSERT_EQ(d_ptr_const_map.h_ncomp_int, d_ptr_map.h_ncomp_int);
    ASSERT_EQ(d_ptr_const_map.h_ncomp_exscan_real,
              d_ptr_map.h_ncomp_exscan_real);
    ASSERT_EQ(d_ptr_const_map.h_ncomp_exscan_int, d_ptr_map.h_ncomp_exscan_int);
    ASSERT_EQ(d_ptr_const_map.ndat_real, d_ptr_map.ndat_real);
    ASSERT_EQ(d_ptr_const_map.ndat_int, d_ptr_map.ndat_int);
    ASSERT_EQ(ncomp_total_real_correct, d_ptr_map.ncomp_total_real);
    ASSERT_EQ(ncomp_total_int_correct, d_ptr_map.ncomp_total_int);
    ASSERT_EQ(d_ptr_const_map.ncomp_total_real, d_ptr_map.ncomp_total_real);
    ASSERT_EQ(d_ptr_const_map.ncomp_total_int, d_ptr_map.ncomp_total_int);

    const std::size_t ndat_real = A->particle_dats_real.size();
    const std::size_t ndat_int = A->particle_dats_int.size();

    ASSERT_EQ(ndat_real, d_ptr_const_map.ndat_real);
    ASSERT_EQ(ndat_int, d_ptr_const_map.ndat_int);

    std::vector<REAL *const *const *> h_ptr_const_real(ndat_real);
    std::vector<INT *const *const *> h_ptr_const_int(ndat_int);
    std::vector<REAL ***> h_ptr_real(ndat_real);
    std::vector<INT ***> h_ptr_int(ndat_int);
    std::vector<int> h_ncomp_real(ndat_real);
    std::vector<int> h_ncomp_int(ndat_int);
    std::vector<int> h_ncomp_exscan_real(ndat_real);
    std::vector<int> h_ncomp_exscan_int(ndat_int);

    sycl_target->queue
        .memcpy(h_ptr_const_real.data(), d_ptr_const_map.d_ptr_real,
                ndat_real * sizeof(REAL *const *const *))
        .wait_and_throw();
    sycl_target->queue
        .memcpy(h_ptr_const_int.data(), d_ptr_const_map.d_ptr_int,
                ndat_int * sizeof(INT *const *const *))
        .wait_and_throw();
    sycl_target->queue
        .memcpy(h_ptr_real.data(), d_ptr_map.d_ptr_real,
                ndat_real * sizeof(REAL ***))
        .wait_and_throw();
    sycl_target->queue
        .memcpy(h_ptr_int.data(), d_ptr_map.d_ptr_int,
                ndat_int * sizeof(INT ***))
        .wait_and_throw();

    sycl_target->queue
        .memcpy(h_ncomp_real.data(), d_ptr_const_map.d_ncomp_real,
                ndat_real * sizeof(int))
        .wait_and_throw();
    sycl_target->queue
        .memcpy(h_ncomp_int.data(), d_ptr_const_map.d_ncomp_int,
                ndat_int * sizeof(int))
        .wait_and_throw();
    sycl_target->queue
        .memcpy(h_ncomp_exscan_real.data(), d_ptr_const_map.d_ncomp_exscan_real,
                ndat_real * sizeof(int))
        .wait_and_throw();
    sycl_target->queue
        .memcpy(h_ncomp_exscan_int.data(), d_ptr_const_map.d_ncomp_exscan_int,
                ndat_int * sizeof(int))
        .wait_and_throw();

    int index = 0;
    int total_ncomp = 0;
    for (auto [sym, dat] : A->particle_dats_real) {
      const int ncomp = dat->ncomp;
      ASSERT_EQ(h_ptr_const_real.at(index), dat->cell_dat.device_ptr());
      ASSERT_EQ(h_ptr_real.at(index), dat->cell_dat.device_ptr());
      ASSERT_EQ(h_ncomp_real.at(index), ncomp);
      ASSERT_EQ(h_ncomp_exscan_real.at(index), total_ncomp);
      ASSERT_EQ(d_ptr_const_map.h_ncomp_real[index], ncomp);
      ASSERT_EQ(d_ptr_const_map.h_ncomp_exscan_real[index], total_ncomp);
      total_ncomp += ncomp;
      index++;
    }
    ASSERT_EQ(d_ptr_const_map.ncomp_total_real, total_ncomp);

    index = 0;
    total_ncomp = 0;
    for (auto [sym, dat] : A->particle_dats_int) {
      const int ncomp = dat->ncomp;
      ASSERT_EQ(h_ptr_const_int.at(index), dat->cell_dat.device_ptr());
      ASSERT_EQ(h_ptr_int.at(index), dat->cell_dat.device_ptr());
      ASSERT_EQ(h_ncomp_int.at(index), ncomp);
      ASSERT_EQ(h_ncomp_exscan_int.at(index), total_ncomp);
      ASSERT_EQ(d_ptr_const_map.h_ncomp_int[index], ncomp);
      ASSERT_EQ(d_ptr_const_map.h_ncomp_exscan_int[index], total_ncomp);
      total_ncomp += ncomp;
      index++;
    }
    ASSERT_EQ(d_ptr_const_map.ncomp_total_int, total_ncomp);
  };

  lambda_test();
  A->remove_particle_dat(Sym<REAL>("FOO"));
  lambda_test();
  A->add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<INT>("BAR"), 4),
                                  A->domain->mesh->get_cell_count()));
  lambda_test();

  sycl_target->free();
}

TEST(ParticleGroup, add_particles_local_particle_group_b) {
  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_2d(100, 16, 32);
  auto A = A_t;
  auto sycl_target = sycl_target_t;
  const int cell_count = A->domain->mesh->get_cell_count();

  auto B = std::make_shared<TestParticleGroup>(A->domain, A->particle_spec,
                                               sycl_target);

  auto pgtmp = std::make_shared<ParticleGroupTemporary>();
  auto C = pgtmp->get(A);

  auto bb = particle_sub_group(
      B, [&](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));
  auto cc = particle_sub_group(
      C, [&](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  ASSERT_TRUE(bb->create_if_required());
  ASSERT_TRUE(cc->create_if_required());

  B->add_particles_local_old(A);
  C->add_particles_local(A);

  ASSERT_TRUE(bb->create_if_required());
  ASSERT_FALSE(bb->create_if_required());
  ASSERT_TRUE(cc->create_if_required());
  ASSERT_FALSE(cc->create_if_required());

  ASSERT_EQ(A->get_npart_local(), B->get_npart_local());
  ASSERT_EQ(A->get_npart_local(), C->get_npart_local());

  auto lambda_test = [&](auto m) {
    for (auto [sym, dat] : m) {
      ASSERT_TRUE(B->contains_dat(sym));
      ASSERT_TRUE(C->contains_dat(sym));
      ASSERT_EQ(dat->ncomp, B->get_dat(sym)->ncomp);
      ASSERT_EQ(dat->ncomp, C->get_dat(sym)->ncomp);
    }
  };

  lambda_test(A->particle_dats_real);
  lambda_test(A->particle_dats_int);

  for (int cx = 0; cx < cell_count; cx++) {
    const int nrow = A->get_npart_cell(cx);
    ASSERT_EQ(nrow, B->get_npart_cell(cx));
    ASSERT_EQ(nrow, C->get_npart_cell(cx));

    auto lambda_test = [&](auto m) {
      for (auto [sym, dat] : m) {
        int tmp;
        sycl_target->queue
            .memcpy(&tmp, B->get_dat(sym)->d_npart_cell + cx, sizeof(int))
            .wait_and_throw();
        ASSERT_EQ(nrow, tmp);
        sycl_target->queue
            .memcpy(&tmp, C->get_dat(sym)->d_npart_cell + cx, sizeof(int))
            .wait_and_throw();
        ASSERT_EQ(nrow, tmp);
        ASSERT_EQ(B->get_dat(sym)->h_npart_cell[cx], nrow);
        ASSERT_EQ(C->get_dat(sym)->h_npart_cell[cx], nrow);
        auto Cd = C->get_cell(sym, cx);
        auto Bd = B->get_cell(sym, cx);
        const int ncomp = dat->ncomp;

        for (int nx = 0; nx < ncomp; nx++) {
          for (int rx = 0; rx < nrow; rx++) {
            ASSERT_EQ(Cd->at(rx, nx), Bd->at(rx, nx));
          }
        }
      }
    };
    lambda_test(A->particle_dats_real);
    lambda_test(A->particle_dats_int);
  }

  B->add_particles_local_old(A);
  C->add_particles_local(A);

  ASSERT_TRUE(bb->create_if_required());
  ASSERT_FALSE(bb->create_if_required());
  ASSERT_TRUE(cc->create_if_required());
  ASSERT_FALSE(cc->create_if_required());

  ASSERT_EQ(2 * A->get_npart_local(), B->get_npart_local());
  ASSERT_EQ(2 * A->get_npart_local(), C->get_npart_local());

  lambda_test(A->particle_dats_real);
  lambda_test(A->particle_dats_int);

  for (int cx = 0; cx < cell_count; cx++) {
    const int nrow = 2 * A->get_npart_cell(cx);
    ASSERT_EQ(nrow, B->get_npart_cell(cx));
    ASSERT_EQ(nrow, C->get_npart_cell(cx));

    auto lambda_test = [&](auto m) {
      for (auto [sym, dat] : m) {
        int tmp;
        sycl_target->queue
            .memcpy(&tmp, B->get_dat(sym)->d_npart_cell + cx, sizeof(int))
            .wait_and_throw();
        ASSERT_EQ(nrow, tmp);
        sycl_target->queue
            .memcpy(&tmp, C->get_dat(sym)->d_npart_cell + cx, sizeof(int))
            .wait_and_throw();
        ASSERT_EQ(nrow, tmp);
        ASSERT_EQ(B->get_dat(sym)->h_npart_cell[cx], nrow);
        ASSERT_EQ(C->get_dat(sym)->h_npart_cell[cx], nrow);
        auto Cd = C->get_cell(sym, cx);
        auto Bd = B->get_cell(sym, cx);
        const int ncomp = dat->ncomp;

        for (int nx = 0; nx < ncomp; nx++) {
          for (int rx = 0; rx < nrow; rx++) {
            ASSERT_EQ(Cd->at(rx, nx), Bd->at(rx, nx));
          }
        }
      }
    };
    lambda_test(A->particle_dats_real);
    lambda_test(A->particle_dats_int);
  }

  B->set_zero_values(3.14, -4);

  B->clear();
  C->clear();

  B->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<REAL>("BAR"), 2), cell_count));
  B->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<INT>("BAR"), 4), cell_count));
  C->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<REAL>("BAR"), 2), cell_count));
  C->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<INT>("BAR"), 4), cell_count));

  B->add_particles_local(A);
  C->add_particles_local(A);

  for (int cx = 0; cx < cell_count; cx++) {
    auto BBAR_real = B->get_cell(Sym<REAL>("BAR"), cx);
    auto BBAR_int = B->get_cell(Sym<INT>("BAR"), cx);
    auto CBAR_real = C->get_cell(Sym<REAL>("BAR"), cx);
    auto CBAR_int = C->get_cell(Sym<INT>("BAR"), cx);
    const int nrow = B->get_npart_cell(cx);
    for (int rx = 0; rx < nrow; rx++) {
      for (int nx = 0; nx < 2; nx++) {
        ASSERT_EQ(BBAR_real->at(rx, nx), 3.14);
        ASSERT_EQ(CBAR_real->at(rx, nx), 0.0);
      }
      for (int nx = 0; nx < 4; nx++) {
        ASSERT_EQ(BBAR_int->at(rx, nx), -4);
        ASSERT_EQ(CBAR_int->at(rx, nx), 0);
      }
    }
  }

  cc = nullptr;
  pgtmp->restore(A, C);
  sycl_target->free();
}

TEST(ParticleGroup, contains_dat_sym_ncomp) {
  auto [A_t, sycl_target_t, cell_count_t] = particle_loop_common_2d(1, 16, 32);
  auto A = A_t;
  auto sycl_target = sycl_target_t;

  ASSERT_TRUE(A->contains_dat(Sym<REAL>("P"), 2));
  ASSERT_FALSE(A->contains_dat(Sym<REAL>("P"), 3));
  ASSERT_TRUE(A->contains_dat(Sym<INT>("NESO_MPI_RANK"), 2));
  ASSERT_FALSE(A->contains_dat(Sym<INT>("NESO_MPI_RANK"), 3));
  ASSERT_FALSE(A->contains_dat(Sym<REAL>("MADE_UP_DAT"), 2));
  ASSERT_FALSE(A->contains_dat(Sym<INT>("MADE_UP_DAT"), 2));

  A->add_particle_dat(Sym<REAL>("MADE_UP_DAT"), 2);
  ASSERT_TRUE(A->contains_dat(Sym<REAL>("MADE_UP_DAT"), 2));
  ASSERT_FALSE(A->contains_dat(Sym<INT>("MADE_UP_DAT"), 2));
  A->add_particle_dat(Sym<INT>("MADE_UP_DAT"), 3);
  ASSERT_TRUE(A->contains_dat(Sym<INT>("MADE_UP_DAT"), 3));

  sycl_target->free();
}
