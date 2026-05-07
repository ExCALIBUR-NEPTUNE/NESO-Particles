#include "include/test_neso_particles.hpp"

namespace {

inline void wrapper_collision_cells(ParticleGroupSharedPtr A,
                                    SYCLTargetSharedPtr sycl_target,
                                    const int cell_count) {

  A->add_particle_dat(Sym<INT>("C"), 1);
  A->add_particle_dat(Sym<INT>("D"), 1);

  auto mesh = std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh);
  const int ndim = mesh->get_ndim();

  auto collision_cells =
      std::make_shared<SubdivideCellsCartesianHMesh>(sycl_target, mesh);

  std::vector<int> num_subdivisions(cell_count);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int s = cellx % 5;
    num_subdivisions.at(cellx) = s;
  }

  collision_cells->set_num_subdivisions(num_subdivisions);
  auto num_collision_cells = collision_cells->get_num_subdivision_cells();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int s = cellx % 5;
    ASSERT_EQ(num_collision_cells.at(cellx), std::pow(std::pow(2, s), ndim));
  }

  auto cdc_origins =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 3, 1);
  auto cdc_num_subdivisions =
      std::make_shared<CellDatConst<int>>(sycl_target, cell_count, 1, 1);

  auto h_origins = cdc_origins->get_all_cells();
  auto h_cdc_num_subdivisions = cdc_num_subdivisions->get_all_cells();
  auto owned_cells = mesh->get_owned_cells();
  const REAL cell_width = mesh->get_cell_width_fine();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int dx = 0; dx < ndim; dx++) {
      h_origins.at(cellx)->at(dx, 0) =
          owned_cells.at(cellx).at(dx) * cell_width;
    }
    h_cdc_num_subdivisions.at(cellx)->at(0, 0) = num_subdivisions.at(cellx);
  }
  cdc_origins->set_all_cells(h_origins);
  cdc_num_subdivisions->set_all_cells(h_cdc_num_subdivisions);

  const REAL k_cell_width_fine = mesh->get_cell_width_fine();

  auto lambda_test = [&](auto B) {
    collision_cells->map(B, Sym<INT>("C"), 0);

    particle_loop(
        B,
        [=](auto P, auto CDC_ORIGIN, auto CDC_NUM_SUBDIVISIONS, auto D) {
          int cells[3];

          const int num_cells_dim = 1 << CDC_NUM_SUBDIVISIONS.at(0, 0);
          const REAL cell_width = k_cell_width_fine / num_cells_dim;
          const REAL inverse_cell_width = 1.0 / cell_width;

          for (int dx = 0; dx < ndim; dx++) {
            const REAL sp = P.at(dx) - CDC_ORIGIN.at(dx, 0);
            const REAL cr = sp * inverse_cell_width;
            const int ci = cr;
            cells[dx] = Kernel::clamp(ci, 0, num_cells_dim - 1);
          }

          int index = cells[ndim - 1];
          for (int dx = ndim - 2; dx >= 0; dx--) {
            index *= num_cells_dim;
            index += cells[dx];
          }
          D.at(0) = index;
        },
        Access::read(Sym<REAL>("P")), Access::read(cdc_origins),
        Access::read(cdc_num_subdivisions), Access::write(Sym<INT>("D")))
        ->execute();

    ErrorPropagate ep(sycl_target);
    auto k_ep = ep.device_ptr();

    particle_loop(
        B,
        [=](auto INDEX, auto C, auto D) {
          NESO_KERNEL_ASSERT(C.at(0) == D.at(0), k_ep);
          const int s = INDEX.cell % 5;
          const int num_subdivisions = 1 << s;
          const int total_num_cells =
              (ndim == 2)
                  ? num_subdivisions * num_subdivisions
                  : num_subdivisions * num_subdivisions * num_subdivisions;
          NESO_KERNEL_ASSERT(C.at(0) >= 0, k_ep);
          NESO_KERNEL_ASSERT(C.at(0) < total_num_cells, k_ep);
        },
        Access::read(ParticleLoopIndex{}), Access::read(Sym<INT>("C")),
        Access::read(Sym<INT>("D")))
        ->execute();

    ASSERT_FALSE(ep.get_flag());
  };

  lambda_test(A);
  lambda_test(particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 1; },
      Access::read(Sym<INT>("ID"))));
}

} // namespace

TEST(CartesianHMesh, collision_cells) {
  const int ncellx = 16;
  const int ncelly = 32;
  const int ncellz = 7;

  {
    auto [A, sycl_target, cell_count] =
        particle_loop_common_2d(15, ncellx, ncelly);

    wrapper_collision_cells(A, sycl_target, cell_count);

    sycl_target->free();
    A->domain->mesh->free();
  }

  {
    auto [A, sycl_target, cell_count] =
        particle_loop_common_3d(15, ncellx, ncelly, ncellz);

    wrapper_collision_cells(A, sycl_target, cell_count);

    sycl_target->free();
    A->domain->mesh->free();
  }
}
