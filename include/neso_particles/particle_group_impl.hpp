#ifndef _NESO_PARTICLES_PARTICLE_GROUP_IMPL_H_
#define _NESO_PARTICLES_PARTICLE_GROUP_IMPL_H_

#include "containers/descendant_products.hpp"
#include "global_mapping.hpp"
#include "loop/particle_loop_iteration_set.hpp"
#include "particle_group.hpp"
#include "particle_sub_group/particle_sub_group.hpp"

namespace NESO::Particles {

template <typename T>
inline void ParticleGroup::add_particle_dat(const Sym<T> sym, const int ncomp) {
  this->add_particle_dat(ParticleDat(this->sycl_target,
                                     ParticleProp(sym, ncomp),
                                     this->domain->mesh->get_cell_count()));
}

inline void ParticleGroup::add_particles() {
  NESOASSERT(false, "Not implemented yet - use add_particles_local and hybrid "
                    "move or parallel advection initialisation.");
};
template <typename U>
inline void ParticleGroup::add_particles([[maybe_unused]] U particle_data) {
  NESOASSERT(false, "Not implemented yet - use add_particles_local and hybrid "
                    "move or parallel advection initialisation.");
};

template <typename T>
inline void ParticleGroup::remove_particles(const int npart, T *usm_cells,
                                            T *usm_layers) {
  this->layer_compressor.remove_particles(npart, usm_cells, usm_layers);
  this->set_npart_cell_from_dat();
  this->invalidate_group_version();
}

inline void ParticleGroup::print_inner(std::ostream &os, SymStore print_spec) {

  os << "==============================================================="
        "================="
     << std::endl;
  for (int cellx = 0; cellx < this->domain->mesh->get_cell_count(); cellx++) {
    if (this->h_npart_cell.ptr[cellx] > 0) {

      std::vector<CellData<REAL>> cell_data_real;
      std::vector<CellData<INT>> cell_data_int;

      int nrow = -1;
      for (auto &symx : print_spec.syms_real) {
        auto cell_data =
            this->particle_dats_real[symx]->cell_dat.get_cell(cellx);
        cell_data_real.push_back(cell_data);
        if (nrow >= 0) {
          NESOASSERT(nrow == cell_data->nrow, "nrow missmatch");
        }
        nrow = cell_data->nrow;
      }
      for (auto &symx : print_spec.syms_int) {
        auto cell_data =
            this->particle_dats_int[symx]->cell_dat.get_cell(cellx);
        cell_data_int.push_back(cell_data);
        if (nrow >= 0) {
          NESOASSERT(nrow == cell_data->nrow, "nrow missmatch");
        }
        nrow = cell_data->nrow;
      }

      os << "------- " << cellx << " -------" << std::endl;
      for (auto &symx : print_spec.syms_real) {
        os << "| " << symx.name << " ";
      }
      for (auto &symx : print_spec.syms_int) {
        os << "| " << symx.name << " ";
      }
      os << "|" << std::endl;

      for (int rowx = 0; rowx < nrow; rowx++) {
        for (auto &cx : cell_data_real) {
          os << "| ";
          for (int colx = 0; colx < cx->ncol; colx++) {
            os << fixed_width_format((*cx)[colx][rowx]) << " ";
          }
        }
        for (auto &cx : cell_data_int) {
          os << "| ";
          for (int colx = 0; colx < cx->ncol; colx++) {
            os << fixed_width_format((*cx)[colx][rowx]) << " ";
          }
        }

        os << "|" << std::endl;
      }
    }
  }

  os << "==============================================================="
        "================="
     << std::endl;
}

inline void ParticleGroup::print(std::ostream &os, SymStore print_spec) {
  this->print_inner(os, print_spec);
}

inline void ParticleGroup::print(SymStore print_spec) {
  this->print_inner(std::cout, print_spec);
}

template <typename... T>
inline void ParticleGroup::print(std::ostream &os, T &&...args) {
  SymStore print_spec(std::forward<T>(args)...);
  this->print_inner(os, print_spec);
}

template <typename... T>
inline void ParticleGroup::print(std::ofstream &os, T &&...args) {
  SymStore print_spec(std::forward<T>(args)...);
  this->print_inner(os, print_spec);
}

template <typename... T> inline void ParticleGroup::print(T &&...args) {
  SymStore print_spec(std::forward<T>(args)...);
  this->print_inner(std::cout, print_spec);
}

inline void ParticleGroup::print_particle(std::ostream &os, const int cell,
                                          const int layer) {
  NESOASSERT(0 <= cell && cell < this->ncell, "Bad input cell.");
  const auto nlayers = this->get_npart_cell(cell);
  NESOASSERT(0 <= layer && layer < nlayers, "Bad input layer.");

  auto lambda_print_dat = [&](auto sym, auto dat) {
    os << "\t" << sym.name << ": ";
    auto data = dat->cell_dat.get_cell(cell);
    auto ncomp = dat->ncomp;
    for (int cx = 0; cx < ncomp; cx++) {
      os << data->at(layer, cx) << " ";
    }
    os << std::endl;
  };

  for (auto d : this->particle_dats_int) {
    lambda_print_dat(d.first, d.second);
  }
  for (auto d : this->particle_dats_real) {
    lambda_print_dat(d.first, d.second);
  }
}

inline void ParticleGroup::print_particle(const int cell, const int layer) {
  this->print_particle(std::cout, cell, layer);
}

typedef std::shared_ptr<ParticleGroup> ParticleGroupSharedPtr;

} // namespace NESO::Particles

#endif
