#ifndef _NESO_PARTICLES_TEST_PARTICLE_SUB_GROUP_H_
#define _NESO_PARTICLES_TEST_PARTICLE_SUB_GROUP_H_

#include "test_neso_particles.hpp"

constexpr int ndim = 2;

inline auto subgroup_test_common(const int N = 1093) {
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
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<INT>("MARKER"), 1)};

  auto A = make_test_obj<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int rank = sycl_target->comm_pair.rank_parent;
  const INT id_offset = rank * N;

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

  return A;
}

class TestParticleSubGroup : public ParticleSubGroup {
public:
  template <typename KERNEL, typename... ARGS>
  TestParticleSubGroup(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                       ARGS... args)
      : ParticleSubGroup(particle_group, kernel, args...) {}

  template <typename KERNEL, typename... ARGS>
  TestParticleSubGroup(ParticleSubGroupSharedPtr particle_group, KERNEL kernel,
                       ARGS... args)
      : ParticleSubGroup(particle_group, kernel, args...) {}

  inline int test_get_cells_layers(std::vector<INT> &cells,
                                   std::vector<INT> &layers) {
    return get_cells_layers(cells, layers);
  }
};

struct TestSubGroupSelector
    : public ParticleSubGroupImplementation::SubGroupSelector {

  template <typename... T>
  TestSubGroupSelector(T... args)
      : ParticleSubGroupImplementation::SubGroupSelector(args...) {}
};

struct TestCellSubGroupSelector
    : public ParticleSubGroupImplementation::CellSubGroupSelector {

  template <typename... T>
  TestCellSubGroupSelector(T... args)
      : ParticleSubGroupImplementation::CellSubGroupSelector(args...) {}
};

template <typename T>
inline bool check_selector(ParticleGroupSharedPtr particle_group,
                           std::shared_ptr<T> selector, std::vector<int> &cells,
                           std::vector<int> &layers) {
  bool status = true;
  ParticleSubGroupImplementation::Selection s;
  selector->create(&s);

  auto lambda_check_eq = [&](auto a, auto b) { status = status && (a == b); };
  auto lambda_check_true = [&](const bool a) { status = status && a; };

  lambda_check_eq(static_cast<std::size_t>(cells.size()), layers.size());
  lambda_check_eq(static_cast<std::size_t>(s.npart_local), layers.size());
  lambda_check_eq(s.ncell, particle_group->domain->mesh->get_cell_count());

  std::map<int, std::set<int>> map_cells_layers;
  for (int ix = 0; ix < s.npart_local; ix++) {
    auto c = cells.at(ix);
    auto l = layers.at(ix);
    lambda_check_true(!static_cast<bool>(map_cells_layers[c].count(l)));
    map_cells_layers[c].insert(l);
  }

  auto sycl_target = particle_group->sycl_target;
  std::vector<int> tmp_int(s.ncell);
  sycl_target->queue
      .memcpy(tmp_int.data(), s.d_npart_cell, s.ncell * sizeof(int))
      .wait_and_throw();

  for (int cx = 0; cx < s.ncell; cx++) {
    lambda_check_eq(static_cast<std::size_t>(s.h_npart_cell[cx]),
                    map_cells_layers[cx].size());
    lambda_check_eq(static_cast<std::size_t>(tmp_int.at(cx)),
                    map_cells_layers[cx].size());
  }

  std::vector<INT> tmp_INT(s.ncell);
  sycl_target->queue
      .memcpy(tmp_INT.data(), s.d_npart_cell_es, s.ncell * sizeof(INT))
      .wait_and_throw();

  INT total = 0;
  for (int cx = 0; cx < s.ncell; cx++) {
    lambda_check_eq(tmp_INT.at(cx), total);
    total += s.h_npart_cell[cx];
  }

  auto map_device =
      ParticleSubGroupImplementation::get_host_map_cells_to_particles(
          sycl_target, s);
  for (int cx = 0; cx < s.ncell; cx++) {
    const int nrow = s.h_npart_cell[cx];
    lambda_check_true(map_device.at(cx).size() ==
                      static_cast<std::size_t>(nrow));
    std::set<int> in_map;
    for (int rx = 0; rx < nrow; rx++) {
      in_map.insert(map_device.at(cx).at(rx));
    }
    lambda_check_eq(in_map, map_cells_layers.at(cx));
  }

  return status;
}

#endif
