#include "include/test_neso_particles.hpp"

template <typename... ARGS>
class ParticlePairLoopArgs : public ParticlePairLoopBase {
protected:
  /// Tuple of the arguments passed to the ParticlePairLoop on construction.
  std::tuple<ARGS...> annotated_args;
  /// Tuple of the arguments with the A/B specification stripped away. These
  /// should be the same types that would be passed to ParticleLoop.
  std::tuple<ARGS...> args;

  /// Recursively assemble the tuple args.
  template <size_t INDEX, typename U> inline void unpack_args(U a0) {
    std::get<INDEX>(this->annotated_args) = a0;
    std::get<INDEX>(this->args) = Access::strip_pair_group_annotation(a0);
  }
  template <size_t INDEX, typename U, typename... V>
  inline void unpack_args(U a0, V... args) {
    std::get<INDEX>(this->annotated_args) = a0;
    std::get<INDEX>(this->args) = Access::strip_pair_group_annotation(a0);
    this->unpack_args<INDEX + 1>(args...);
  }

public:
  std::string name;
  virtual ~ParticlePairLoopArgs<ARGS...>() = default;

  ParticlePairLoopArgs<ARGS...>(std::string &name, ARGS... args) : name(name) {
    this->unpack_args<0>(args...);
  }
};

template <typename GROUP_TYPE> struct CellwisePairListAbsolute;

template <> struct CellwisePairListAbsolute<ParticleGroup> {
  ParticleGroupSharedPtr A;
  ParticleGroupSharedPtr B;
  DSMC::CellwisePairListSharedPtr pair_list;
  CellwisePairListAbsolute<ParticleGroup>() = default;
  ~CellwisePairListAbsolute<ParticleGroup>() = default;

  CellwisePairListAbsolute<ParticleGroup>(
      ParticleGroupSharedPtr A, ParticleGroupSharedPtr B,
      DSMC::CellwisePairListSharedPtr pair_list)
      : A(A), B(B), pair_list(pair_list) {}
};

template <typename KERNEL, typename... ARGS>
class ParticlePairLoopCellwisePairList : public ParticlePairLoopArgs<ARGS...> {

protected:
public:
  std::vector<CellwisePairListAbsolute<ParticleGroup>> pair_lists;
  ParticlePairLoopKernel<KERNEL> kernel;

  ParticlePairLoopCellwisePairList(
      std::string name,
      std::vector<CellwisePairListAbsolute<ParticleGroup>> pair_lists,
      ParticlePairLoopKernel<KERNEL> kernel, ARGS... args)
      : ParticlePairLoopArgs<ARGS...>(name, args...), pair_lists(pair_lists),
        kernel(kernel) {}

  virtual inline void
  execute([[maybe_unused]] const std::optional<int> cell_start = std::nullopt,
          [[maybe_unused]] const std::optional<int> cell_end =
              std::nullopt) override {}

  virtual inline void
  submit([[maybe_unused]] const std::optional<int> cell_start = std::nullopt,
         [[maybe_unused]] const std::optional<int> cell_end =
             std::nullopt) override {}

  virtual inline void wait() override {}
};

TEST(ParticlePairLoop, base) {

  const int npart_cell = 10;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 32;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NUM_NEIGHBOURS"), 1);

  auto cellwise_pair_listA =
      std::make_shared<DSMC::CellwisePairList>(sycl_target, cell_count);

  // TODO create some pairs here

  ParticlePairLoopCellwisePairList pl0(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup>(A, A, cellwise_pair_listA)},
      ParticlePairLoopKernel{[](auto NN_A, auto NN_B) {
        NN_A.at(0)++;
        NN_B.at(0)++;
      }},
      Access::A(Access::write(Sym<INT>("NUM_NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NUM_NEIGHBOURS"))));

  pl0.execute();

  sycl_target->free();
}
