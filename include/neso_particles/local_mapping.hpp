#ifndef _NESO_PARTICLES_LOCAL_MAPPING
#define _NESO_PARTICLES_LOCAL_MAPPING

#include <memory>
#include <mpi.h>

#include "compute_target.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

// Forward declaration of ParticleGroup
class ParticleGroup;

/**
 * Base class to map positions to owning MPI ranks based on local
 * information, i.e. halo cells. Could also use or set the cell id.
 */
class LocalMapper {
public:
  /**
   * This function maps particle positions to cells on the underlying mesh.
   *
   * @param particle_group ParticleGroup containing particle positions.
   * @param map_cell Optionally map a particular cell. If map_cell is 0 then an
   * implementation may assume that the call is in the second part of
   * hybrid_move and particles must be binned into a cell.
   */
  virtual inline void map(ParticleGroup &particle_group,
                          const int map_cell = -1) = 0;

  /**
   * Callback for ParticleGroup to execute for additional setup of the
   * LocalMapper that may involve the ParticleGroup.
   *
   * @param particle_group ParticleGroup instance.
   */
  virtual inline void
  particle_group_callback(ParticleGroup &particle_group) = 0;

  /**
   * This function is called by the ParticleGroup in cell_move.
   *
   * @param particle_group ParticleGroup containing particle positions.
   * @param map_cell Optionally map a particular cell.
   */
  virtual inline void map_cells([[maybe_unused]] ParticleGroup &particle_group,
                                [[maybe_unused]] const int map_cell = -1) {};
};

typedef std::shared_ptr<LocalMapper> LocalMapperSharedPtr;

/**
 *  Dummy LocalMapper implementation that does nothing to use as a default.
 */
class DummyLocalMapperT : public LocalMapper {
private:
public:
  /// Disable (implicit) copies.
  DummyLocalMapperT(const DummyLocalMapperT &st) = delete;
  /// Disable (implicit) copies.
  DummyLocalMapperT &operator=(DummyLocalMapperT const &a) = delete;

  virtual ~DummyLocalMapperT(){};

  /**
   *  No-op Constructor.
   */
  DummyLocalMapperT(){};

  /**
   *  No-op implementation of map.
   */
  inline void map([[maybe_unused]] ParticleGroup &particle_group,
                  [[maybe_unused]] const int map_cell = -1) {};

  /**
   *  No-op implementation of callback.
   */
  inline void
  particle_group_callback([[maybe_unused]] ParticleGroup &particle_group) {};
};

inline std::shared_ptr<DummyLocalMapperT> DummyLocalMapper() {
  return std::make_shared<DummyLocalMapperT>();
}

} // namespace NESO::Particles
#endif
