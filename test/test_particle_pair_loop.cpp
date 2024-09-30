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
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 2),
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

} // namespace


class CellPairIterationSet {
public:
  /**
   * There are 3 cases:
   *  1) Cell i with cell i which requires masking off particles with themselves.
   *  2) Cell i with cell j both owned by this rank.
   *  3) Cell i with halo cell j which requires communication from a remote rank.
   */
  struct DeviceNeighbourMap {
    // Are cells considered neighbours of themselves.
    int cell_with_self;

  };

  DeviceNeighbourMap device_neighbour_map;
  DomainSharedPtr domain;

  CellPairIterationSet(
    DomainSharedPtr domain
  ) : 
    domain(domain) 
  {

  }

  virtual inline DeviceNeighbourMap get_device_neighbour_map() {
    return this->device_neighbour_map;
  }

  virtual inline DomainSharedPtr get_domain(){
    return this->domain;
  }


};

class CellSelfIterationSet : public CellPairIterationSet {
public:
  CellSelfIterationSet(
    DomainSharedPtr domain
  ) : 
    CellPairIterationSet(domain) 
  {
    this->device_neighbour_map.cell_with_self = 1;
  }

};



class ParticlePairLoopBase {
public:
  /**
   *  Execute the particle loop and block until completion. Must be called
   *  Collectively on the communicator.
   */
  virtual inline void execute(const std::optional<int> cell = std::nullopt) = 0;

  /**
   *  Launch the ParticleLoop and return. Must be called collectively over the
   *  MPI communicator of the ParticleGroup. Loop execution is complete when
   *  the corresponding call to wait returns.
   */
  virtual inline void submit(const std::optional<int> cell = std::nullopt) = 0;

  /**
   * Wait for loop execution to complete. On completion perform post-loop
   * actions. Must be called collectively on communicator.
   */
  virtual inline void wait() = 0;
};


template <typename GROUP_A_TYPE, typename GROUP_B_TYPE, typename KERNEL, typename... ARGS>
class ParticlePairLoopCellWise : public ParticlePairLoopBase {};

template <typename KERNEL, typename... ARGS>
class ParticlePairLoopCellWise<ParticleGroup, ParticleGroup, KERNEL, ARGS...> : public ParticlePairLoopBase {
protected:
  std::shared_ptr<CellPairIterationSet> cell_pair_iteration_set;
  std::shared_ptr<ParticleGroup> group_a;
  std::shared_ptr<ParticleGroup> group_b;
  KERNEL kernel;
  
public:

  ParticlePairLoopCellWise(
    std::shared_ptr<CellPairIterationSet> cell_pair_iteration_set,
    std::shared_ptr<ParticleGroup> group_a,
    std::shared_ptr<ParticleGroup> group_b,
    KERNEL kernel,
    ARGS... args
  ) :
    cell_pair_iteration_set(cell_pair_iteration_set),
    group_a(group_a),
    group_b(group_b),
    kernel(kernel)
  {

  }

  virtual inline void execute(const std::optional<int> cell = std::nullopt) override{

  }
  virtual inline void submit(const std::optional<int> cell = std::nullopt) override{

  }
  virtual inline void wait() override{

  }

};


template <typename GROUP_A_TYPE, typename GROUP_B_TYPE, typename KERNEL, typename... ARGS>
[[nodiscard]] auto particle_pair_loop(
    std::shared_ptr<CellPairIterationSet> cell_pair_iteration_set,
    std::shared_ptr<GROUP_A_TYPE> group_a,
    std::shared_ptr<GROUP_B_TYPE> group_b,
    KERNEL kernel,
    ARGS... args
){
  auto loop = std::make_shared<ParticlePairLoopCellWise<GROUP_A_TYPE, GROUP_B_TYPE, KERNEL, ARGS...>>(
    cell_pair_iteration_set,
    group_a,
    group_b,
    kernel,
    args...
  );
  return std::dynamic_pointer_cast<ParticlePairLoopBase>(loop);
}


TEST(ParticlePairLoop, base) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto pair_loop = particle_pair_loop(
    std::make_shared<CellSelfIterationSet>(A->domain),
    A,
    A,
    [=](auto P){

    },
    Access::read(Sym<REAL>("P"))
  );

  A->free();
  sycl_target->free();
  mesh->free();
}

