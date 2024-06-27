#ifndef _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_COMMON_HPP_
#define _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_COMMON_HPP_

#include "../../../loop/particle_loop.hpp"
#include "../../../particle_sub_group.hpp"
#include "../dmplex_interface.hpp"
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace NESO::Particles::PetscInterface {

/**
 * TODO
 */
class BoundaryInteractionCommon {
protected:
  inline ParticleGroupSharedPtr
  get_particle_group(ParticleGroupSharedPtr iteration_set) {
    return iteration_set;
  }
  inline ParticleGroupSharedPtr
  get_particle_group(ParticleSubGroupSharedPtr iteration_set) {
    return iteration_set->get_particle_group();
  }

  template <typename T>
  inline void check_dat(ParticleGroupSharedPtr particle_group, Sym<T> sym,
                        const int ncomp) {
    if (!particle_group->contains_dat(sym)) {
      ParticleProp prop(sym, ncomp);
      particle_group->add_particle_dat(
          ParticleDat(this->sycl_target, prop, this->mesh->get_cell_count()));
    } else {
      NESOASSERT(
          particle_group->get_dat(sym)->ncomp >= ncomp,
          "Requested dat with sym " + sym.name +
              " exists already with an insufficient number of components");
    }
  }

  template <typename T>
  inline void prepare_particle_group(std::shared_ptr<T> particle_sub_group) {
    auto particle_group = this->get_particle_group(particle_sub_group);
    NESOASSERT(particle_group->sycl_target == this->sycl_target,
               "Missmatch of sycl targets.");
    const int ndim = this->mesh->get_ndim();
    this->check_dat(particle_group, this->previous_position_sym, ndim);
    this->check_dat(particle_group, this->boundary_position_sym, ndim);
    this->check_dat(particle_group, this->boundary_label_sym, 3);
  }

  inline std::set<PetscInt> get_labels() const {
    std::map<PetscInt, std::set<PetscInt>> bl;
    for (auto &item : this->boundary_groups) {
      for (auto ix : item.second) {
        bl[item.first].insert(ix);
      }
    }

    std::set<PetscInt> labels;
    for (auto &item : bl) {
      for (auto ix : item.second) {
        NESOASSERT(labels.count(ix) == 0, "Label " + std::to_string(ix) +
                                              " exists in the specification of "
                                              "more than one boundary group.");
        labels.insert(ix);
      }
    }
    return labels;
  }

public:
  SYCLTargetSharedPtr sycl_target;
  DMPlexInterfaceSharedPtr mesh;
  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;

  Sym<REAL> previous_position_sym;
  Sym<REAL> boundary_position_sym;
  Sym<INT> boundary_label_sym;

  BoundaryInteractionCommon(
      SYCLTargetSharedPtr sycl_target, DMPlexInterfaceSharedPtr mesh,
      std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
      std::optional<Sym<REAL>> previous_position_sym = std::nullopt,
      std::optional<Sym<REAL>> boundary_position_sym = std::nullopt,
      std::optional<Sym<INT>> boundary_label_sym = std::nullopt)
      : sycl_target(sycl_target), mesh(mesh), boundary_groups(boundary_groups) {
    auto assign_sym = [=](auto &output_sym, auto &input_sym, auto default_sym) {
      if (input_sym != std::nullopt) {
        output_sym = input_sym.value();
      } else {
        output_sym = default_sym;
      }
    };
    assign_sym(this->previous_position_sym, previous_position_sym,
               Sym<REAL>("NESO_PARTICLES_DMPLEX_BOUNDARY_PREV_POS"));
    assign_sym(this->boundary_position_sym, boundary_position_sym,
               Sym<REAL>("NESO_PARTICLES_DMPLEX_BOUNDARY_POS"));
    assign_sym(this->boundary_label_sym, boundary_label_sym,
               Sym<INT>("NESO_PARTICLES_DMPLEX_BOUNDARY_LABEL"));
  }

  /**
   * TODO
   */
  template <typename T>
  inline void pre_integration(std::shared_ptr<T> particle_sub_group) {
    prepare_particle_group(particle_sub_group);
    auto particle_group = this->get_particle_group(particle_sub_group);
    auto position_dat = particle_group->position_dat;
    const int k_ncomp = position_dat->ncomp;
    const int k_ndim = this->mesh->get_ndim();
    NESOASSERT(
        k_ncomp >= k_ndim,
        "Positions ncomp is smaller than the number of mesh dimensions.");

    particle_loop(
        "BoundaryInteractionCommon::pre_integration", particle_sub_group,
        [=](auto P, auto PP) {
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            PP.at(dimx) = P.at(dimx);
          }
        },
        Access::read(position_dat->sym),
        Access::write(this->previous_position_sym))
        ->execute();
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
