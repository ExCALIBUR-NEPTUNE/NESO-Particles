#ifndef _NESO_PARTICLES_CELL_DAT_MOVE
#define _NESO_PARTICLES_CELL_DAT_MOVE

#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include "cell_dat.hpp"
#include "cell_dat_compression.hpp"
#include "compute_target.hpp"
#include "error_propagate.hpp"
#include "particle_dat.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "profiling.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"
#include <iomanip>

namespace NESO::Particles {

/**
 *  CellMove is the implementation that moves particles between cells. When a
 *  particle moves to a new cell the corresponding data is moved to the new
 *  cell in all the ParticleDat instances.
 */
class CellMove {

#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#else
protected:
#endif
  ParticleDatSharedPtr<INT> cell_id_dat;

  // layer compressor from the ParticleGroup for removing the old particle rows
  LayerCompressor &layer_compressor;

  // ErrorPropagate object to detect bad cell indices
  ErrorPropagate ep_bad_cell_indices;

  inline void print_particle(const int cell, const int layer) {
    nprint("Particle info, cell:", cell, "layer:", layer);
    std::cout << std::setprecision(18);
    auto lambda_print_dat = [&](auto sym, auto dat) {
      std::cout << "\t" << sym.name << ": ";
      auto data = dat->cell_dat.get_cell(cell);
      auto ncomp = dat->ncomp;
      for (int cx = 0; cx < ncomp; cx++) {
        std::cout << data->at(layer, cx) << " ";
      }
      std::cout << std::endl;
    };

    for (auto d : *this->particle_group_pointer_map->particle_dats_int) {
      lambda_print_dat(d.first, d.second);
    }
    for (auto d : *this->particle_group_pointer_map->particle_dats_real) {
      lambda_print_dat(d.first, d.second);
    }
  }

  ParticleGroupPointerMapSharedPtr particle_group_pointer_map{nullptr};

public:
  /// Disable (implicit) copies.
  CellMove(const CellMove &st) = delete;
  /// Disable (implicit) copies.
  CellMove &operator=(CellMove const &a) = delete;

  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;

  ~CellMove() {}
  /**
   * Create a cell move instance to move particles between cells.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param layer_compressor LayerCompressor to use to compress ParticleDat
   * instances.
   * @param particle_dats_real Container of REAL ParticleDat.
   * @param particle_dats_int Container of INT ParticleDat.
   */
  CellMove(SYCLTargetSharedPtr sycl_target, LayerCompressor &layer_compressor,
           ParticleGroupPointerMapSharedPtr particle_group_pointer_map)
      : layer_compressor(layer_compressor), ep_bad_cell_indices(sycl_target),
        particle_group_pointer_map(particle_group_pointer_map),
        sycl_target(sycl_target) {}

  /**
   * Set the ParticleDat to use as a source for cell ids.
   *
   * @param cell_id_dat ParticleDat to use for cell ids.
   */
  inline void set_cell_id_dat(ParticleDatSharedPtr<INT> cell_id_dat) {
    this->cell_id_dat = cell_id_dat;
  }

  /**
   * Move particles between cells (on this MPI rank) using the cell ids on
   * the particles.
   */
  void move();
};

} // namespace NESO::Particles

#endif
