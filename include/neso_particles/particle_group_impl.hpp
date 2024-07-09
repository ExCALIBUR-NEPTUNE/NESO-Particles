#ifndef _NESO_PARTICLES_PARTICLE_GROUP_IMPL_H_
#define _NESO_PARTICLES_PARTICLE_GROUP_IMPL_H_

#include "containers/descendant_products.hpp"
#include "global_mapping.hpp"
#include "particle_group.hpp"
#include "particle_sub_group.hpp"

namespace NESO::Particles {

inline void
ParticleGroup::add_particle_dat(ParticleDatSharedPtr<REAL> particle_dat) {
  if (!this->existing_compatible_dat(particle_dat)) {
    this->particle_dats_real[particle_dat->sym] = particle_dat;
    // Does this dat hold particle positions?
    if (particle_dat->positions) {
      this->position_dat = particle_dat;
      this->position_sym = std::make_shared<Sym<REAL>>(particle_dat->sym.name);
    }
    add_particle_dat_common(particle_dat);
  }
}
inline void
ParticleGroup::add_particle_dat(ParticleDatSharedPtr<INT> particle_dat) {
  if (!this->existing_compatible_dat(particle_dat)) {
    this->particle_dats_int[particle_dat->sym] = particle_dat;
    // Does this dat hold particle cell ids?
    if (particle_dat->positions) {
      this->cell_id_dat = particle_dat;
      this->cell_id_sym = std::make_shared<Sym<INT>>(particle_dat->sym.name);
    }
    add_particle_dat_common(particle_dat);
  }
}

inline void ParticleGroup::add_particles() {
  NESOASSERT(false, "Not implemented yet - use add_particles_local and hybrid "
                    "move or parallel advection initialisation.");
};
template <typename U>
inline void ParticleGroup::add_particles(U particle_data) {
  NESOASSERT(false, "Not implemented yet - use add_particles_local and hybrid "
                    "move or parallel advection initialisation.");
};

/*
 * Number of bytes required to store the data for one particle.
 */
inline size_t ParticleGroup::particle_size() {
  size_t s = 0;
  for (auto &dat : this->particle_dats_real) {
    s += dat.second->cell_dat.row_size();
  }
  for (auto &dat : this->particle_dats_int) {
    s += dat.second->cell_dat.row_size();
  }
  return s;
};

inline void ParticleGroup::add_particles_local(ParticleSet &particle_data) {
  // loop over the cells of the new particles and allocate more space in the
  // dats

  const int npart_new = particle_data.npart;
  auto cellids = particle_data.get(*this->cell_id_sym);
  std::vector<INT> layers(npart_new);
  for (int px = 0; px < npart_new; px++) {
    auto cellindex = cellids[px];
    NESOASSERT((cellindex >= 0) && (cellindex < this->ncell),
               "Bad particle cellid)");

    layers[px] = this->h_npart_cell.ptr[cellindex]++;
  }
  buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait_and_throw();
  this->recompute_npart_cell_es();

  EventStack es;

  for (auto &dat : this->particle_dats_real) {
    realloc_dat(dat.second);
    dat.second->append_particle_data(npart_new,
                                     particle_data.contains(dat.first), cellids,
                                     layers, particle_data.get(dat.first), es);
  }

  for (auto &dat : this->particle_dats_int) {
    realloc_dat(dat.second);
    dat.second->append_particle_data(npart_new,
                                     particle_data.contains(dat.first), cellids,
                                     layers, particle_data.get(dat.first), es);
  }

  es.wait();
  this->check_dats_and_group_agree();
  this->invalidate_group_version();
}

inline void
ParticleGroup::add_particles_local(ParticleSetSharedPtr particle_data) {
  add_particles_local(*particle_data);
}

template <typename T>
inline void ParticleGroup::remove_particles(const int npart, T *usm_cells,
                                            T *usm_layers) {
  this->layer_compressor.remove_particles(npart, usm_cells, usm_layers);
  this->set_npart_cell_from_dat();
  this->invalidate_group_version();
}

inline void ParticleGroup::remove_particles(const int npart,
                                            const std::vector<INT> &cells,
                                            const std::vector<INT> &layers) {

  this->d_remove_cells.realloc_no_copy(npart);
  this->d_remove_layers.realloc_no_copy(npart);

  auto k_cells = this->d_remove_cells.ptr;
  auto k_layers = this->d_remove_layers.ptr;

  NESOASSERT(cells.size() >= npart, "Bad cells length compared to npart");
  NESOASSERT(layers.size() >= npart, "Bad layers length compared to npart");

  auto b_cells = sycl::buffer<INT>(cells.data(), sycl::range<1>(npart));
  auto b_layers = sycl::buffer<INT>(layers.data(), sycl::range<1>(npart));

  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        auto a_cells = b_cells.get_access<sycl::access::mode::read>(cgh);
        auto a_layers = b_layers.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for<>(sycl::range<1>(static_cast<size_t>(npart)),
                           [=](sycl::id<1> idx) {
                             k_cells[idx] = static_cast<INT>(a_cells[idx]);
                             k_layers[idx] = static_cast<INT>(a_layers[idx]);
                           });
      })
      .wait_and_throw();

  this->remove_particles(npart, this->d_remove_cells.ptr,
                         this->d_remove_layers.ptr);
}

inline void ParticleGroup::remove_particles(
    std::shared_ptr<ParticleSubGroup> particle_sub_group) {

  NESOASSERT(particle_sub_group->particle_group.get() == this,
             "remove_particles called with a ParticleSubGroup which is not a "
             "sub group of the ParticleGroup");

  if (particle_sub_group->is_entire_particle_group()) {
    this->clear();
  } else {
    const auto npart = particle_sub_group->get_npart_local();
    this->d_remove_cells.realloc_no_copy(npart);
    this->d_remove_layers.realloc_no_copy(npart);
    auto k_cells = this->d_remove_cells.ptr;
    auto k_layers = this->d_remove_layers.ptr;
    particle_sub_group->get_cells_layers(k_cells, k_layers);
    this->remove_particles(npart, k_cells, k_layers);
  }
}

/*
 * Perform global move operation to send particles to the MPI ranks stored in
 * the first component of the NESO_MPI_RANK ParticleDat. Must be called
 * collectively on the communicator.
 */
inline void ParticleGroup::global_move() {
  this->global_move_ctx.move();
  this->set_npart_cell_from_dat();
  this->invalidate_group_version();
}

/*
 * Perform local move operation to send particles to the MPI ranks stored in
 * the second component of the NESO_MPI_RANK ParticleDat. Must be called
 * collectively on the communicator.
 */
inline void ParticleGroup::local_move() {
  this->local_move_ctx->move();
  this->set_npart_cell_from_dat();
  this->invalidate_group_version();
}

inline std::string fixed_width_format(INT value) {
  char buffer[128];
  const int err = snprintf(buffer, 128, "%lld", static_cast<long long>(value));
  NESOASSERT(err >= 0 && err < 128, "Bad snprintf return code.");
  return std::string(buffer);
}
inline std::string fixed_width_format(REAL value) {
  char buffer[128];
  const int err = snprintf(buffer, 128, "%f", value);
  NESOASSERT(err >= 0 && err < 128, "Bad snprintf return code.");
  return std::string(buffer);
}

template <typename... T> inline void ParticleGroup::print(T... args) {

  SymStore print_spec(args...);

  std::cout << "==============================================================="
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

      std::cout << "------- " << cellx << " -------" << std::endl;
      for (auto &symx : print_spec.syms_real) {
        std::cout << "| " << symx.name << " ";
      }
      for (auto &symx : print_spec.syms_int) {
        std::cout << "| " << symx.name << " ";
      }
      std::cout << "|" << std::endl;

      for (int rowx = 0; rowx < nrow; rowx++) {
        for (auto &cx : cell_data_real) {
          std::cout << "| ";
          for (int colx = 0; colx < cx->ncol; colx++) {
            std::cout << fixed_width_format((*cx)[colx][rowx]) << " ";
          }
        }
        for (auto &cx : cell_data_int) {
          std::cout << "| ";
          for (int colx = 0; colx < cx->ncol; colx++) {
            std::cout << fixed_width_format((*cx)[colx][rowx]) << " ";
          }
        }

        std::cout << "|" << std::endl;
      }
    }
  }

  std::cout << "==============================================================="
               "================="
            << std::endl;
}

/*
 * Perform global move operation. Uses particle positions to determine
 * global/neighbour MPI ranks and uses the global then local move methods (in
 * that order). Must be called collectively on the communicator.
 */
inline void ParticleGroup::hybrid_move() {
  reset_mpi_ranks(this->mpi_rank_dat);
  this->domain->local_mapper->map(*this);
  this->mesh_hierarchy_global_map->execute();

  ProfileRegion r_global("hybrid_move", "global_move");
  this->global_move_ctx.move();
  this->set_npart_cell_from_dat();
  r_global.end();
  sycl_target->profile_map.add_region(r_global);

  this->domain->local_mapper->map(*this, 0);

  ProfileRegion r_local("hybrid_move", "local_move");
  this->local_move_ctx->move();
  r_local.end();
  sycl_target->profile_map.add_region(r_local);

  this->set_npart_cell_from_dat();
  this->invalidate_group_version();
}

/**
 * Clear all particles from the ParticleGroup on the calling MPI rank.
 */
inline void ParticleGroup::clear() {
  const int cell_count = this->domain->mesh->get_cell_count();
  for (int cx = 0; cx < cell_count; cx++) {
    this->h_npart_cell.ptr[cx] = 0;
  }
  buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait_and_throw();
  this->recompute_npart_cell_es();
  this->realloc_all_dats();
  EventStack es;
  for (auto &dat : this->particle_dats_real) {
    es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }
  for (auto &dat : this->particle_dats_int) {
    es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }
  es.wait();
  this->check_dats_and_group_agree();
  this->invalidate_group_version();
}

inline void ParticleGroup::add_particles_local(
    std::shared_ptr<ProductMatrix> product_matrix) {
  this->add_particles_local(product_matrix, nullptr, nullptr, nullptr);
}

inline void ParticleGroup::add_particles_local(
    std::shared_ptr<DescendantProducts> descendant_products,
    std::shared_ptr<ParticleGroup> source_particle_group) {

  ParticleGroup *source_particle_group_ptr =
      source_particle_group != nullptr ? source_particle_group.get() : nullptr;

  this->add_particles_local(
      std::dynamic_pointer_cast<ProductMatrix>(descendant_products),
      descendant_products->d_parent_cells->ptr,
      descendant_products->d_parent_layers->ptr, source_particle_group_ptr);
}

inline void ParticleGroup::add_particles_local(
    std::shared_ptr<ProductMatrix> product_matrix, const INT *d_cells,
    const INT *d_layers, ParticleGroup *source_particle_group) {

  NESOASSERT(((d_layers == nullptr) && (d_cells == nullptr)) ||
                 ((d_layers != nullptr) && (d_cells != nullptr)),
             "d_cells and d_layers must either both be nullptr or both be not "
             "nullptr.");

  NESOASSERT(this->particle_spec.contains(product_matrix->spec->particle_spec),
             "A particle property was passed to add_particles_local which does "
             "not exist in the ParticleGroup.");

  const int num_products = product_matrix->num_products;
  // reuse this space
  this->d_remove_cells.realloc_no_copy(num_products);
  this->d_remove_layers.realloc_no_copy(num_products);
  INT *cells_ptr = this->d_remove_cells.ptr;
  INT *layers_ptr = this->d_remove_layers.ptr;
  auto d_pm = product_matrix->impl_get_const();

  // Either the cells are set in the product matrix or we need to default
  // initialise them to 0.
  const auto cell_id_sym = this->cell_id_dat->sym;
  const int cell_id_index = product_matrix->spec->get_sym_index(cell_id_sym);

  // If cells not explicitly defined in the ProductMatrix.
  if (cell_id_index == -1) {
    if (d_cells == nullptr) { // If no parent cells use cell 0.
      this->sycl_target->queue.fill(cells_ptr, (INT)0, num_products)
          .wait_and_throw();
    } else { // Copy parent cells.
      this->sycl_target->queue
          .memcpy(cells_ptr, d_cells, sizeof(INT) * num_products)
          .wait_and_throw();
    }
  } else { // Use cells explicitly defined in the product matrix
    INT const *source_cells_ptr = d_pm.ptr_int + num_products * cell_id_index;
    this->sycl_target->queue
        .memcpy(cells_ptr, source_cells_ptr, sizeof(INT) * num_products)
        .wait_and_throw();
  }

  // If any of the cells are -1 we trim those particles by setting layers to -1.
  // Get the new cell occupancies at the destination layers of the particles to
  // insert
  // get dst layers also sets h_npart_cell, d_npart_cell
  this->get_new_layers(num_products, cells_ptr, layers_ptr);
  this->recompute_npart_cell_es();

  // Allocate space in all the dats for the new particles
  this->realloc_all_dats();

  // copy the particle data into the particle dats
  // for each particle property either there is data in the product matrix or
  // it should be initalised to 0
  EventStack es;

  // lambda which copies the property
  auto lambda_copy = [&](auto dat, const auto index, const auto d_offsets,
                         const auto d_src) {
    const int ncomp_local = dat->ncomp;
    const int ncomp_matrix = product_matrix->spec->get_num_components(dat->sym);
    NESOASSERT(ncomp_local == ncomp_matrix,
               "Missmatch in the number of components in the product matrix "
               "and the number of components on the particle group for a sym.");

    auto dat_ptr = dat->impl_get();
    const int k_ncomp = dat->ncomp;
    const int k_nproducts = num_products;
    es.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(
          sycl::range<1>(static_cast<size_t>(num_products)),
          [=](sycl::id<1> idx) {
            const INT cell = cells_ptr[idx];
            const INT layer = layers_ptr[idx];
            // layer == -1 implies the entry is invalid somehow
            if (layer > -1) {
              for (int nx = 0; nx < k_ncomp; nx++) {
                dat_ptr[cell][nx][layer] =
                    d_src[(d_offsets[index] + nx) * k_nproducts + idx];
              }
            }
          });
    }));
  };

  // lambda to copy property from parent particle
  auto lambda_parent_copy = [&](auto dat, auto dat_src) {
    auto dat_ptr = dat->impl_get();
    const auto dat_src_ptr = dat_src->impl_get_const();
    const int k_ncomp = dat->ncomp;
    es.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(sycl::range<1>(static_cast<size_t>(num_products)),
                         [=](sycl::id<1> idx) {
                           const INT cell = cells_ptr[idx];
                           const INT layer = layers_ptr[idx];
                           const INT cell_src = d_cells[idx];
                           const INT layer_src = d_layers[idx];
                           // layer == -1 implies the entry is invalid somehow
                           if (layer > -1) {
                             for (int nx = 0; nx < k_ncomp; nx++) {
                               dat_ptr[cell][nx][layer] =
                                   dat_src_ptr[cell_src][nx][layer_src];
                             }
                           }
                         });
    }));
  };

  // dispatch to the copy lambda or zero lambda depending on if the data exists
  auto lambda_dispatch = [&](auto dat, const auto d_offsets, const auto d_src) {
    const int index = product_matrix->spec->get_sym_index(dat->sym);
    // if the property is not in the ProductMatrix
    if (index == -1) {
      if (d_cells == nullptr) { // Zero properties if no parents defined
        zero_dat_properties(dat, num_products, cells_ptr, layers_ptr, es);
      } else { // Else copy the property from the parent
        // Is there a separate parent
        if (source_particle_group != nullptr) {
          // Does this dat exist in the separate parent
          if (source_particle_group->contains_dat(dat->sym)) {
            auto dat_src = source_particle_group->get_dat(dat->sym);
            lambda_parent_copy(dat, dat_src);
          } else {
            NESOWARN(false, "Child ParticleGroup has a particle property which "
                            "does not exist on the parent ParticleGroup.");
            zero_dat_properties(dat, num_products, cells_ptr, layers_ptr, es);
          }
        } else {
          lambda_parent_copy(dat, dat);
        }
      }
    } else { // Copy from the ProductMatrix
      lambda_copy(dat, index, d_offsets, d_src);
    }
  };

  // launch the copies into the ParticleDats
  for (auto &dat : this->particle_dats_real) {
    lambda_dispatch(dat.second, d_pm.offsets_real, d_pm.ptr_real);
  }
  for (auto &dat : this->particle_dats_int) {
    lambda_dispatch(dat.second, d_pm.offsets_int, d_pm.ptr_int);
  }
  // launch the copies of the npart cells into the ParticleDats
  for (auto &dat : this->particle_dats_real) {
    es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }
  for (auto &dat : this->particle_dats_int) {
    es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }

  // wait for the copy kernels
  es.wait();
  this->check_dats_and_group_agree();
  this->invalidate_group_version();
}

inline void ParticleGroup::add_particles_local(
    std::shared_ptr<ParticleGroup> particle_group) {

  // Get the new cell occupancies
  const int cell_count = this->domain->mesh->get_cell_count();
  std::vector<INT> h_npart_cell_existing(cell_count);
  std::vector<INT> h_npart_cell_to_add(cell_count);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    h_npart_cell_existing.at(cellx) = this->h_npart_cell.ptr[cellx];
    const INT to_add = particle_group->h_npart_cell.ptr[cellx];
    this->h_npart_cell.ptr[cellx] += to_add;
    h_npart_cell_to_add.at(cellx) = to_add;
  }
  buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait_and_throw();
  this->recompute_npart_cell_es();

  // Allocate space in all the dats for the new particles
  this->realloc_all_dats();
  EventStack es;

  auto lambda_dispatch = [&](auto dat_dst) {
    // get a shared ptr of the right type
    auto dat_src = dat_dst;
    dat_src = nullptr;
    auto sym = dat_dst->sym;
    if (particle_group->contains_dat(sym)) {
      dat_src = particle_group->get_dat(sym);
    }
    dat_dst->append_particle_data(dat_src, h_npart_cell_existing,
                                  h_npart_cell_to_add, es);
  };

  // launch the copies of the npart cells into the ParticleDats
  for (auto &dat : this->particle_dats_real) {
    lambda_dispatch(dat.second);
  }
  for (auto &dat : this->particle_dats_int) {
    lambda_dispatch(dat.second);
  }

  // launch the copies of the npart cells into the ParticleDats
  for (auto &dat : this->particle_dats_real) {
    es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }
  for (auto &dat : this->particle_dats_int) {
    es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }
  es.wait();
  this->invalidate_group_version();
}

inline void ParticleGroup::add_particles_local(
    std::shared_ptr<ParticleSubGroup> particle_sub_group) {

  if (particle_sub_group->is_entire_particle_group()) {
    this->add_particles_local(particle_sub_group->particle_group);
  } else {
    const INT npart = particle_sub_group->get_npart_local();
    this->d_remove_cells.realloc_no_copy(npart);
    this->d_remove_layers.realloc_no_copy(2 * npart);
    auto k_cells = this->d_remove_cells.ptr;
    auto k_layers_src = this->d_remove_layers.ptr;
    auto k_layers_dst = k_layers_src + npart;
    particle_sub_group->get_cells_layers(k_cells, k_layers_src);
    // get dst layers also sets h_npart_cell, d_npart_cell
    this->get_new_layers(npart, k_cells, k_layers_dst);
    this->recompute_npart_cell_es();
    // Allocate space in all the dats for the new particles
    this->realloc_all_dats();

    EventStack es;

    // lambda which copies the property
    auto lambda_copy = [&](auto dat, const auto dat_src) {
      const int ncomp_dst = dat->ncomp;
      const int ncomp_src = dat_src->ncomp;
      NESOASSERT(
          ncomp_dst == ncomp_src,
          "Missmatch in the number of components in the local ParticleGroup"
          "and the number of components on the ParticleGroup for the "
          "ParticleSubGroup.");

      auto dst_ptr = dat->impl_get();
      auto src_ptr = dat_src->impl_get_const();
      const int k_ncomp = dat->ncomp;
      const int k_npart = npart;
      es.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(
            sycl::range<1>(static_cast<size_t>(k_npart)), [=](sycl::id<1> idx) {
              const INT cell = k_cells[idx];
              const INT src_layer = k_layers_src[idx];
              const INT dst_layer = k_layers_dst[idx];
              for (int nx = 0; nx < k_ncomp; nx++) {
                dst_ptr[cell][nx][dst_layer] = src_ptr[cell][nx][src_layer];
              }
            });
      }));
    };

    auto lambda_dispatch = [&](auto dat) {
      auto sym = dat->sym;
      if (particle_sub_group->particle_group->contains_dat(sym)) {
        lambda_copy(dat, particle_sub_group->particle_group->get_dat(sym));
      } else {
        zero_dat_properties(dat, npart, k_cells, k_layers_dst, es);
      }
    };

    // launch the copies into the ParticleDats
    for (auto &dat : this->particle_dats_real) {
      lambda_dispatch(dat.second);
    }
    for (auto &dat : this->particle_dats_int) {
      lambda_dispatch(dat.second);
    }
    // launch the copies of the npart cells into the ParticleDats
    for (auto &dat : this->particle_dats_real) {
      es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
    }
    for (auto &dat : this->particle_dats_int) {
      es.push(dat.second->async_set_npart_cells(this->h_npart_cell));
    }

    es.wait();
    this->check_dats_and_group_agree();
    this->invalidate_group_version();
  }
}

typedef std::shared_ptr<ParticleGroup> ParticleGroupSharedPtr;

} // namespace NESO::Particles

#endif
