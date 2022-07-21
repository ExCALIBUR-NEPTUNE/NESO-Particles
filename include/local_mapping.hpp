#ifndef _NESO_PARTICLES_LOCAL_MAPPING
#define _NESO_PARTICLES_LOCAL_MAPPING

#include <CL/sycl.hpp>
#include <mpi.h>

#include "compute_target.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

/*
 * Abstract class to map positions to owning MPI ranks based on local
 * information, i.e. halo cells. Could also use or set the cell id.
 */
class LocalMapper {
public:
  inline void map(ParticleDatShPtr<REAL> &position_dat,
                  ParticleDatShPtr<INT> &cell_id_dat,
                  ParticleDatShPtr<INT> &mpi_rank_dat);
};

typedef std::shared_ptr<LocalMapper> LocalMapperShPtr;

/*
 *  Dummy LocalMapper implementation that does nothing to use as a default.
 */
class DummyLocalMapperT : public LocalMapper {
private:
public:
  inline void map(ParticleDatShPtr<REAL> &position_dat,
                  ParticleDatShPtr<INT> &cell_id_dat,
                  ParticleDatShPtr<INT> &mpi_rank_dat){};
};

inline LocalMapperShPtr DummyLocalMapper() {
  return std::make_shared<DummyLocalMapperT>();
}

} // namespace NESO::Particles
#endif
