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

  sycl_target->free();
}
