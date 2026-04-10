#include "include/test_neso_particles.hpp"

namespace {
void voronoi_cell_test_wrapper(ParticleGroupSharedPtr A,
                               SYCLTargetSharedPtr sycl_target,
                               const int cell_count) {

  const int ndim = A->domain->mesh->get_ndim();
  A->add_particle_dat(Sym<INT>("FOO"), 3);

  auto lambda_reset = [&]() {
    particle_loop(
        A,
        [=](auto FOO) {
          FOO.at(0) = -1;
          FOO.at(1) = -1;
        },
        Access::write(Sym<INT>("FOO")))
        ->execute();
  };

  auto voronoi_points =
      std::make_shared<CellDat<REAL>>(sycl_target, cell_count, ndim);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  // Test when no Voronoi cells are specified that the result is the 0-th cell.
  {
    SubdivideCellsVoronoi mapper(sycl_target, voronoi_points);

    lambda_reset();
    mapper.map(A, Sym<INT>("FOO"), 1);

    particle_loop(
        A,
        [=](auto FOO) {
          NESO_KERNEL_ASSERT(FOO.at(0) == -1, k_ep);
          NESO_KERNEL_ASSERT(FOO.at(1) == 0, k_ep);
        },
        Access::read(Sym<INT>("FOO")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());

    lambda_reset();

    auto aa = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));

    auto bb = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 1; },
        Access::read(Sym<INT>("ID")));

    mapper.map(aa, Sym<INT>("FOO"), 1);
    particle_loop(
        aa,
        [=](auto FOO) {
          NESO_KERNEL_ASSERT(FOO.at(0) == -1, k_ep);
          NESO_KERNEL_ASSERT(FOO.at(1) == 0, k_ep);
        },
        Access::read(Sym<INT>("FOO")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
    particle_loop(
        bb,
        [=](auto FOO) {
          NESO_KERNEL_ASSERT(FOO.at(0) == -1, k_ep);
          NESO_KERNEL_ASSERT(FOO.at(1) == -1, k_ep);
        },
        Access::read(Sym<INT>("FOO")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  }

  // Set some points for each mesh cell.
  {

    const int max_num_vcells = (ndim == 2) ? 7 : 3;

    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int nrow = cellx % max_num_vcells;
      voronoi_points->set_nrow(cellx, nrow);
    }
    voronoi_points->wait_set_nrow();

    std::mt19937 rng(522342 + sycl_target->comm_pair.rank_parent);
    std::uniform_real_distribution<REAL> dist(0.0, 32.0);

    std::vector<CellData<REAL>> h_cell_data;
    h_cell_data.reserve(cell_count);

    EventStack es;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int nrow = cellx % max_num_vcells;
      auto m = std::make_shared<CellDataT<REAL>>(sycl_target, nrow, ndim);

      for (int rx = 0; rx < nrow; rx++) {
        for (int cx = 0; cx < ndim; cx++) {
          m->at(rx, cx) = dist(rng);
        }
      }

      h_cell_data.push_back(m);
      voronoi_points->set_cell_async(cellx, m, es);
    }

    es.wait();

    SubdivideCellsVoronoi mapper(sycl_target, voronoi_points);
    lambda_reset();
    mapper.map(A, Sym<INT>("FOO"), 1);

    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto P = A->get_cell(Sym<REAL>("P"), cellx);
      auto FOO = A->get_cell(Sym<INT>("FOO"), cellx);
      const int nrow = P->nrow;
      const int npoints = cellx % max_num_vcells;
      for (int rowx = 0; rowx < nrow; rowx++) {

        int vcell = 0;
        REAL min_dist = std::numeric_limits<REAL>::max();

        for (int pointx = 0; pointx < npoints; pointx++) {
          REAL dist = 0.0;
          for (int dx = 0; dx < ndim; dx++) {
            const REAL r =
                P->at(rowx, dx) - h_cell_data.at(cellx)->at(pointx, dx);
            const REAL r2 = r * r;
            dist += r2;
          }
          if (dist < min_dist) {
            vcell = pointx;
            min_dist = dist;
          }
        }

        const int to_test = static_cast<int>(FOO->at(rowx, 1));
        ASSERT_EQ(vcell, to_test);
        ASSERT_EQ(FOO->at(rowx, 0), -1);
      }
    }

    particle_loop(
        A,
        [=](auto FOO) {
          FOO.at(2) = FOO.at(1);
          FOO.at(0) = -1;
          FOO.at(1) = -1;
        },
        Access::write(Sym<INT>("FOO")))
        ->execute();

    auto aa = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));

    auto bb = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 1; },
        Access::read(Sym<INT>("ID")));

    lambda_reset();
    particle_loop(
        A,
        [=](auto FOO) {
          NESO_KERNEL_ASSERT(FOO.at(0) == -1, k_ep);
          NESO_KERNEL_ASSERT(FOO.at(1) == -1, k_ep);
        },
        Access::read(Sym<INT>("FOO")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());

    mapper.map(aa, Sym<INT>("FOO"), 1);

    particle_loop(
        aa,
        [=](auto FOO) {
          NESO_KERNEL_ASSERT(FOO.at(0) == -1, k_ep);
          NESO_KERNEL_ASSERT(FOO.at(1) == FOO.at(2), k_ep);
        },
        Access::read(Sym<INT>("FOO")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
    particle_loop(
        bb,
        [=](auto FOO) {
          NESO_KERNEL_ASSERT(FOO.at(0) == -1, k_ep);
          NESO_KERNEL_ASSERT(FOO.at(1) == -1, k_ep);
        },
        Access::read(Sym<INT>("FOO")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  }
}

} // namespace

TEST(Algorithms, subdivide_cells_voronoi_2d) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);
  voronoi_cell_test_wrapper(A, sycl_target, cell_count_t);
  sycl_target->free();
  A->domain->mesh->free();
}

TEST(Algorithms, subdivide_cells_voronoi_3d) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_3d(15, 16, 3, 32);
  voronoi_cell_test_wrapper(A, sycl_target, cell_count_t);
  sycl_target->free();
  A->domain->mesh->free();
}

namespace {

void cartesian_cell_test_wrapper(ParticleGroupSharedPtr A,
                                 SYCLTargetSharedPtr sycl_target,
                                 const int cell_count) {

  const int ndim = A->domain->mesh->get_ndim();
  A->add_particle_dat(Sym<INT>("FOO"), 3);

  auto lambda_reset = [&]() {
    particle_loop(
        A,
        [=](auto FOO) {
          FOO.at(0) = -1;
          FOO.at(1) = -1;
        },
        Access::write(Sym<INT>("FOO")))
        ->execute();
  };

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  const int max_num_subdivisions = 7;
  std::vector<int> h_num_subdivions(cell_count);
  std::fill(h_num_subdivions.begin(), h_num_subdivions.end(), 1);

  // Test when no subdivisions are specified that the result is the 0-th cell.
  {
    SubdivideCartesianCells mapper(
        sycl_target, std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
        h_num_subdivions);

    lambda_reset();
    mapper.map(A, Sym<INT>("FOO"), 1);

    particle_loop(
        A,
        [=](auto FOO) {
          NESO_KERNEL_ASSERT(FOO.at(0) == -1, k_ep);
          NESO_KERNEL_ASSERT(FOO.at(1) == 0, k_ep);
        },
        Access::read(Sym<INT>("FOO")))
        ->execute();

    ASSERT_FALSE(ep.get_flag());

    lambda_reset();

    auto aa = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));

    auto bb = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 1; },
        Access::read(Sym<INT>("ID")));

    mapper.map(aa, Sym<INT>("FOO"), 1);
    particle_loop(
        aa,
        [=](auto FOO) {
          NESO_KERNEL_ASSERT(FOO.at(0) == -1, k_ep);
          NESO_KERNEL_ASSERT(FOO.at(1) == 0, k_ep);
        },
        Access::read(Sym<INT>("FOO")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
    particle_loop(
        bb,
        [=](auto FOO) {
          NESO_KERNEL_ASSERT(FOO.at(0) == -1, k_ep);
          NESO_KERNEL_ASSERT(FOO.at(1) == -1, k_ep);
        },
        Access::read(Sym<INT>("FOO")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  }

  // Set some points for each mesh cell.
  {}
}

} // namespace

TEST(Algorithms, subdivide_cells_cartesian_2d) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);
  cartesian_cell_test_wrapper(A, sycl_target, cell_count_t);
  sycl_target->free();
  A->domain->mesh->free();
}

TEST(Algorithms, subdivide_cells_cartesian_3d) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_3d(15, 16, 3, 32);
  cartesian_cell_test_wrapper(A, sycl_target, cell_count_t);
  sycl_target->free();
  A->domain->mesh->free();
}
