#ifndef _NESO_PARTICLES_LOCAL_MAPPING
#define _NESO_PARTICLES_LOCAL_MAPPING

#include <CL/sycl.hpp>
#include <memory>
#include <mpi.h>

#include "compute_target.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

/**
 * Abstract class to map positions to owning MPI ranks based on local
 * information, i.e. halo cells. Could also use or set the cell id.
 */
class LocalMapper {
public:
  /**
   * This function maps particle positions to cells on the underlying mesh.
   *
   * @param position_dat ParticleDat containing particle positions.
   * @param cell_id_dat ParticleDat containing particle cell ids.
   * @param mpi_rank_dat ParticleDat containing particle MPI ranks.
   */
  virtual inline void map(ParticleDatShPtr<REAL> &position_dat,
                          ParticleDatShPtr<INT> &cell_id_dat,
                          ParticleDatShPtr<INT> &mpi_rank_dat) = 0;
};

typedef std::shared_ptr<LocalMapper> LocalMapperShPtr;

/**
 *  Dummy LocalMapper implementation that does nothing to use as a default.
 */
class DummyLocalMapperT : public LocalMapper {
private:
public:
  /**
   *  No-op implementation of map.
   */
  inline void map(ParticleDatShPtr<REAL> &position_dat,
                  ParticleDatShPtr<INT> &cell_id_dat,
                  ParticleDatShPtr<INT> &mpi_rank_dat){};
};

inline std::shared_ptr<DummyLocalMapperT> DummyLocalMapper() {
  return std::make_shared<DummyLocalMapperT>();
}

} // namespace NESO::Particles
#endif
