#include <neso_particles/common_impl.hpp>
#include <neso_particles/particle_group.hpp>
#include <neso_particles/particle_group_impl.hpp>

namespace NESO::Particles {

void ParticleGroup::get_new_layers(const int npart,
                                   const INT *RESTRICT cells_ptr,
                                   INT *RESTRICT layers_ptr) {

  if (npart > 0) {
    buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait_and_throw();
    INT *k_npart_cell = this->d_npart_cell.ptr;
    const INT k_ncell = this->ncell;
    this->sycl_target->queue
        .parallel_for(sycl::range<1>(static_cast<size_t>(npart)),
                      [=](sycl::id<1> idx) {
                        const INT cell = cells_ptr[idx];
                        if ((-1 < cell) && (cell < k_ncell)) {
                          sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                                           sycl::memory_scope::device>
                              element_atomic(k_npart_cell[cell]);
                          const INT layer = element_atomic.fetch_add((INT)1);
                          layers_ptr[idx] = layer;
                        } else {
                          layers_ptr[idx] = -1;
                        }
                      })
        .wait_and_throw();
    buffer_memcpy(this->h_npart_cell, this->d_npart_cell).wait_and_throw();
  }
}

bool ParticleGroup::check_validation(ParticleDatVersionTracker &to_check,
                                     const bool update_to_check) {
  bool updated = false;
  for (auto &item : to_check) {
    const auto &key = item.first;
    const int64_t to_check_value = item.second;
    const auto local_entry = this->particle_dat_versions.at(key);
    const int64_t local_value = std::get<0>(local_entry);
    const bool local_bool = std::get<1>(local_entry);
    if ((this->debug_sub_group_indent >= 4) &&
        ((local_value != to_check_value) || (local_bool))) {
      std::cout << std::string(this->debug_sub_group_indent, ' ')
                << "Sym name: " << key.get_sym_name()
                << " version_update: " << (local_value != to_check_value)
                << " always_update: " << local_bool << std::endl;
    }
    // If a local count is different to the count on the to_check then an
    // updated is required on the object that holds to check.
    if (local_value != to_check_value) {
      updated = true;
      if (update_to_check) {
        to_check.at(key) = local_value;
      }
    }
    // If this bool has been set then an update is always required.
    if (local_bool) {
      updated = true;
    }
  }
  return updated;
}

bool ParticleGroup::check_validation(ParticleGroupVersion &to_check,
                                     const bool update_to_check) {
  const bool updated = this->particle_group_version != to_check;
  if (update_to_check && updated) {
    to_check = this->particle_group_version;
  }
  return updated;
}

void ParticleGroup::setup_internal(DomainSharedPtr domain,
                                   ParticleSpec &particle_spec,
                                   SYCLTargetSharedPtr sycl_target) {
  // create the pointer map
  this->particle_group_pointer_map = std::make_shared<ParticleGroupPointerMap>(
      this->sycl_target, &this->particle_dats_real, &this->particle_dats_int);

  this->debug_sub_group_create =
      get_env_size_t("NESO_PARTICLES_DEBUG_SUB_GROUPS", 0);

  this->resource_stack_sub_group_resource =
      std::make_shared<ResourceStack<SubGroupSelectorResource>>(
          std::make_shared<SubGroupSelectorResourceStackInterface>(
              this->sycl_target, this->ncell));

  this->sym_vector_pointer_cache_dispatch =
      std::make_shared<SymVectorPointerCacheDispatch>(
          this->sycl_target, &this->particle_dats_int, nullptr,
          &this->particle_dats_real, nullptr);

  this->h_npart_cell.realloc_no_copy(this->ncell);
  this->d_npart_cell.realloc_no_copy(this->ncell);

  this->dh_npart_cell_es =
      std::make_shared<BufferDeviceHost<INT>>(sycl_target, this->ncell);

  for (int cellx = 0; cellx < this->ncell; cellx++) {
    this->h_npart_cell.ptr[cellx] = 0;
    this->dh_npart_cell_es->h_buffer.ptr[cellx] = 0;
  }
  buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait();
  this->dh_npart_cell_es->host_to_device();

  // Create a ParticleDat to store the MPI rank of the particles in.
  mpi_rank_sym = std::make_shared<Sym<INT>>("NESO_MPI_RANK");
  mpi_rank_dat =
      ParticleDat(sycl_target, ParticleProp(*mpi_rank_sym, 2), ncell);
  add_particle_dat(mpi_rank_dat);
  this->global_move_ctx.set_mpi_rank_dat(mpi_rank_dat);

  // Add the user added dats
  for (auto &property : particle_spec.properties_real) {
    add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
  }
  for (auto &property : particle_spec.properties_int) {
    add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
  }

  this->local_move_ctx = std::make_unique<LocalMove>(
      sycl_target, layer_compressor, particle_dats_real, particle_dats_int,
      domain->mesh->get_local_communication_neighbours().size(),
      domain->mesh->get_local_communication_neighbours().data());
  this->local_move_ctx->set_mpi_rank_dat(mpi_rank_dat);

  this->layer_compressor.set_cell_id_dat(this->cell_id_dat);
  this->cell_move_ctx.set_cell_id_dat(this->cell_id_dat);

  this->mesh_hierarchy_global_map = std::make_shared<MeshHierarchyGlobalMap>(
      this->sycl_target, this->domain->mesh, this->position_dat,
      this->cell_id_dat, this->mpi_rank_dat);

  // call the callback on the local mapper to complete the setup of that
  // object
  this->domain->local_mapper->particle_group_callback(*this);
}

ParticleSetSharedPtr
ParticleGroup::get_particles(const std::size_t num_particles,
                             const INT *const d_cells,
                             const INT *const d_layers) {
  if (num_particles > 0) {
    auto r0 = this->sycl_target->profile_map.start_region("ParticleGroup",
                                                          "get_particles_0");
    auto dh_real = get_resource<BufferDeviceHost<REAL>,
                                ResourceStackInterfaceBufferDeviceHost<REAL>>(
        sycl_target->resource_stack_map,
        ResourceStackKeyBufferDeviceHost<REAL>{}, sycl_target);
    auto dh_int = get_resource<BufferDeviceHost<INT>,
                               ResourceStackInterfaceBufferDeviceHost<INT>>(
        sycl_target->resource_stack_map,
        ResourceStackKeyBufferDeviceHost<INT>{}, sycl_target);

    const auto k_dat_info = this->particle_group_pointer_map->get_const();
    const std::size_t num_elements_real =
        num_particles * static_cast<std::size_t>(k_dat_info.ncomp_total_real);
    const std::size_t num_elements_int =
        num_particles * static_cast<std::size_t>(k_dat_info.ncomp_total_int);
    dh_real->realloc_no_copy(num_elements_real);
    dh_int->realloc_no_copy(num_elements_int);

    REAL *k_real = dh_real->d_buffer.ptr;
    INT *k_int = dh_int->d_buffer.ptr;

    auto e0 = this->sycl_target->queue.parallel_for(
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(k_dat_info.ndat_real, num_particles)),
        [=](sycl::id<2> idx) {
          const INT index_dat = static_cast<INT>(idx[0]);
          const INT index_particle = static_cast<INT>(idx[1]);
          const INT cell = d_cells[index_particle];
          const INT layer = d_layers[index_particle];
          const int ncomp = k_dat_info.d_ncomp_real[index_dat];
          const int ncomp_es = k_dat_info.d_ncomp_exscan_real[index_dat];
          REAL *dst_dat = k_real + ncomp_es * num_particles;
          for (int cx = 0; cx < ncomp; cx++) {
            REAL *dst = dst_dat + cx * num_particles;
            dst[index_particle] =
                k_dat_info.d_ptr_real[index_dat][cell][cx][layer];
          }
        });

    auto e1 = this->sycl_target->queue.parallel_for(
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(k_dat_info.ndat_int, num_particles)),
        [=](sycl::id<2> idx) {
          const INT index_dat = static_cast<INT>(idx[0]);
          const INT index_particle = static_cast<INT>(idx[1]);
          const INT cell = d_cells[index_particle];
          const INT layer = d_layers[index_particle];
          const int ncomp = k_dat_info.d_ncomp_int[index_dat];
          const int ncomp_es = k_dat_info.d_ncomp_exscan_int[index_dat];
          INT *dst_dat = k_int + ncomp_es * num_particles;
          for (int cx = 0; cx < ncomp; cx++) {
            INT *dst = dst_dat + cx * num_particles;
            dst[index_particle] =
                k_dat_info.d_ptr_int[index_dat][cell][cx][layer];
          }
        });

    auto ps = std::make_shared<ParticleSet>(num_particles, this->particle_spec);

    e0.wait_and_throw();
    dh_real->device_to_host(num_elements_real);
    e1.wait_and_throw();
    dh_int->device_to_host(num_elements_int);

    for (int dx = 0; dx < k_dat_info.ndat_real; dx++) {
      const int dat_ncomp = k_dat_info.h_ncomp_real[dx];
      const int dat_ncomp_exscan = k_dat_info.h_ncomp_exscan_real[dx];
      auto sym = this->particle_group_pointer_map->map_index_to_sym_real.at(dx);
      REAL *dst_ptr = ps->get_ptr(sym, 0, 0);
      REAL *src_ptr = dh_real->h_buffer.ptr + dat_ncomp_exscan * num_particles;
      std::memcpy(dst_ptr, src_ptr, num_particles * dat_ncomp * sizeof(REAL));
    }

    for (int dx = 0; dx < k_dat_info.ndat_int; dx++) {
      const int dat_ncomp = k_dat_info.h_ncomp_int[dx];
      const int dat_ncomp_exscan = k_dat_info.h_ncomp_exscan_int[dx];
      auto sym = this->particle_group_pointer_map->map_index_to_sym_int.at(dx);
      INT *dst_ptr = ps->get_ptr(sym, 0, 0);
      INT *src_ptr = dh_int->h_buffer.ptr + dat_ncomp_exscan * num_particles;
      std::memcpy(dst_ptr, src_ptr, num_particles * dat_ncomp * sizeof(INT));
    }

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDeviceHost<REAL>{}, dh_real);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDeviceHost<INT>{}, dh_int);
    this->sycl_target->profile_map.end_region(r0);
    return ps;
  } else {
    return std::make_shared<ParticleSet>(0, this->particle_spec);
  }
}

/**
 * Clear all particles from the ParticleGroup on the calling MPI rank.
 */
void ParticleGroup::clear() {
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

void ParticleGroup::add_particle_dat(ParticleDatSharedPtr<REAL> particle_dat) {
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
void ParticleGroup::add_particle_dat(ParticleDatSharedPtr<INT> particle_dat) {
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

void ParticleGroup::add_particles_local(
    std::shared_ptr<ProductMatrix> product_matrix) {
  this->add_particles_local(product_matrix, nullptr, nullptr, nullptr);
}

void ParticleGroup::add_particles_local(
    std::shared_ptr<ProductMatrix> product_matrix, const INT *d_cells,
    const INT *d_layers, ParticleGroup *source_particle_group) {

  ProfileRegion pr0("ParticleGroup", "add_particles_local_product_matrix");

  NESOASSERT(((d_layers == nullptr) && (d_cells == nullptr)) ||
                 ((d_layers != nullptr) && (d_cells != nullptr)),
             "d_cells and d_layers must either both be nullptr or both be not "
             "nullptr.");

  NESOASSERT(this->particle_spec.contains(product_matrix->spec->particle_spec),
             "A particle property was passed to add_particles_local which does "
             "not exist in the ParticleGroup.");

  const int num_products = product_matrix->num_products;
  if (num_products == 0) {
    this->invalidate_group_version();
    return;
  }

  auto d_buffer =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_buffer->realloc_no_copy(2 * num_products);

  // reuse this space
  INT *cells_ptr = d_buffer->ptr;
  INT *layers_ptr = cells_ptr + num_products;
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

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_buffer);
  pr0.end();
  this->sycl_target->profile_map.add_region(pr0);
}

void ParticleGroup::add_particles_local(
    std::shared_ptr<DescendantProducts> descendant_products,
    std::shared_ptr<ParticleGroup> source_particle_group) {

  ParticleGroup *source_particle_group_ptr =
      source_particle_group != nullptr ? source_particle_group.get() : nullptr;

  this->add_particles_local(
      std::dynamic_pointer_cast<ProductMatrix>(descendant_products),
      descendant_products->d_parent_cells->ptr,
      descendant_products->d_parent_layers->ptr, source_particle_group_ptr);
}

void ParticleGroup::add_particles_local(
    std::shared_ptr<ParticleGroup> particle_group) {

  ProfileRegion pr0("ParticleGroup", "add_particles_local_particle_group");

  NESOASSERT(particle_group.get() != this,
             "Cannot add a ParticleGroup to itself.");

  NESOASSERT(this->domain->mesh->get_cell_count() ==
                 particle_group->domain->mesh->get_cell_count(),
             "Miss-match in cell counts between source and destination "
             "ParticleGroups.");

  auto lambda_test_dat = [&](auto m) {
    for (auto &[sym, dat] : m) {
      if (particle_group->contains_dat(sym)) {
        const int ncomp_correct = dat->ncomp;
        const int ncomp_to_test = particle_group->get_dat(sym)->ncomp;
        NESOASSERT(ncomp_correct == ncomp_to_test,
                   "Dat " + sym.name +
                       "Has a different number of components in source and "
                       "destination ParticleGroups.");
      }
    }
  };
  lambda_test_dat(this->particle_dats_real);
  lambda_test_dat(this->particle_dats_int);

  const auto k_src_map_ptrs =
      particle_group->particle_group_pointer_map->get_const();
  const auto k_dst_map_ptrs = this->particle_group_pointer_map->get();

  // Get the new cell occupancies
  const int cell_count = this->domain->mesh->get_cell_count();
  std::vector<INT> h_npart_cell_existing(cell_count);
  INT total_to_add = 0;
  const INT npart_local_old = this->get_npart_local();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    h_npart_cell_existing.at(cellx) = this->h_npart_cell.ptr[cellx];
    NESOASSERT(h_npart_cell_existing.at(cellx) >= 0, "Bad npart cell.");
    const INT to_add = particle_group->h_npart_cell.ptr[cellx];
    total_to_add += to_add;
    NESOASSERT(to_add >= 0, "Bad npart cell.");
    this->h_npart_cell.ptr[cellx] += to_add;
  }
  NESOASSERT(total_to_add == particle_group->get_npart_local(),
             "Source ParticleGroup is not self consistent.");
  buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait_and_throw();
  this->recompute_npart_cell_es();
  NESOASSERT(npart_local_old + total_to_add == this->get_npart_local(),
             "Destination ParticleGroup is not self consistent.");

  // Allocate space in all the dats for the new particles
  this->realloc_all_dats();

  std::vector<int> src_dat_index_real;
  std::vector<int> dst_dat_index_real;
  std::vector<int> dst_dat_zero_index_real;
  int index = 0;
  for (auto [sym, dat] : this->particle_dats_real) {
    if (particle_group->contains_dat(sym)) {
      src_dat_index_real.push_back(
          particle_group->particle_group_pointer_map->map_sym_to_index_real.at(
              sym));
      dst_dat_index_real.push_back(index);
    } else {
      dst_dat_zero_index_real.push_back(index);
    }
    index++;
  }
  std::vector<int> src_dat_index_int;
  std::vector<int> dst_dat_index_int;
  std::vector<int> dst_dat_zero_index_int;
  index = 0;
  for (auto [sym, dat] : this->particle_dats_int) {
    if (particle_group->contains_dat(sym)) {
      src_dat_index_int.push_back(
          particle_group->particle_group_pointer_map->map_sym_to_index_int.at(
              sym));
      dst_dat_index_int.push_back(index);
    } else {
      dst_dat_zero_index_int.push_back(index);
    }
    index++;
  }

  auto d_npart_cell_existing = std::make_shared<BufferDevice<INT>>(
      this->sycl_target, h_npart_cell_existing);
  auto d_src_dat_index_real = std::make_shared<BufferDevice<int>>(
      this->sycl_target, src_dat_index_real);
  auto d_src_dat_index_int =
      std::make_shared<BufferDevice<int>>(this->sycl_target, src_dat_index_int);
  auto d_dst_dat_index_real = std::make_shared<BufferDevice<int>>(
      this->sycl_target, dst_dat_index_real);
  auto d_dst_dat_index_int =
      std::make_shared<BufferDevice<int>>(this->sycl_target, dst_dat_index_int);
  auto d_dst_dat_zero_index_real = std::make_shared<BufferDevice<int>>(
      this->sycl_target, dst_dat_zero_index_real);
  auto d_dst_dat_zero_index_int = std::make_shared<BufferDevice<int>>(
      this->sycl_target, dst_dat_zero_index_int);

  auto k_npart_cell_existing = d_npart_cell_existing->ptr;
  auto k_src_dat_index_real = d_src_dat_index_real->ptr;
  auto k_src_dat_index_int = d_src_dat_index_int->ptr;
  auto k_dst_dat_index_real = d_dst_dat_index_real->ptr;
  auto k_dst_dat_index_int = d_dst_dat_index_int->ptr;
  auto k_dst_dat_zero_index_real = d_dst_dat_zero_index_real->ptr;
  auto k_dst_dat_zero_index_int = d_dst_dat_zero_index_int->ptr;

  const std::size_t k_num_real_copy = dst_dat_index_real.size();
  const std::size_t k_num_int_copy = dst_dat_index_int.size();
  const std::size_t k_num_real_zero = dst_dat_zero_index_real.size();
  const std::size_t k_num_int_zero = dst_dat_zero_index_int.size();

  EventStack es;

  // Create a ParticleLoop iteration set to loop over all the particles in the
  // source ParticleGroup.
  ParticleLoopImplementation::ParticleLoopBlockIterationSet iteration_set(
      particle_group->mpi_rank_dat);
  particle_group->check_dats_and_group_agree();

  NESOASSERT(particle_group->get_dat(Sym<INT>("NESO_MPI_RANK")).get() ==
                 particle_group->mpi_rank_dat.get(),
             "Bad MPI_RANK_DAT");

  const std::size_t nbin =
      this->sycl_target->parameters->template get<SizeTParameter>("LOOP_NBIN")
          ->value;
  const std::size_t local_size =
      this->sycl_target->parameters
          ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;
  auto is = iteration_set.get_all_cells(nbin, local_size, 0);

  const auto k_zero_value_real = this->zero_value_real;
  const auto k_zero_value_int = this->zero_value_int;

  for (auto &blockx : is) {
    const auto block_device = blockx.block_device;
    es.push(sycl_target->queue.parallel_for<>(
        blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
          std::size_t cell;
          std::size_t src_layer;
          block_device.get_cell_layer(idx, &cell, &src_layer);
          const std::size_t dst_layer = k_npart_cell_existing[cell] + src_layer;
          if (block_device.work_item_required(cell, src_layer)) {
            for (std::size_t dx = 0; dx < k_num_real_copy; dx++) {
              const int dst_dat_index = k_dst_dat_index_real[dx];
              const int src_dat_index = k_src_dat_index_real[dx];
              const int ncomp = k_dst_map_ptrs.d_ncomp_real[dst_dat_index];
              auto dst_ptrs = k_dst_map_ptrs.d_ptr_real[dst_dat_index];
              auto src_ptrs = k_src_map_ptrs.d_ptr_real[src_dat_index];
              for (int cx = 0; cx < ncomp; cx++) {
                dst_ptrs[cell][cx][dst_layer] = src_ptrs[cell][cx][src_layer];
              }
            }
            for (std::size_t dx = 0; dx < k_num_int_copy; dx++) {
              const int dst_dat_index = k_dst_dat_index_int[dx];
              const int src_dat_index = k_src_dat_index_int[dx];
              const int ncomp = k_dst_map_ptrs.d_ncomp_int[dst_dat_index];
              auto dst_ptrs = k_dst_map_ptrs.d_ptr_int[dst_dat_index];
              auto src_ptrs = k_src_map_ptrs.d_ptr_int[src_dat_index];
              for (int cx = 0; cx < ncomp; cx++) {
                dst_ptrs[cell][cx][dst_layer] = src_ptrs[cell][cx][src_layer];
              }
            }
            for (std::size_t dx = 0; dx < k_num_real_zero; dx++) {
              const int dst_dat_index = k_dst_dat_zero_index_real[dx];
              const int ncomp = k_dst_map_ptrs.d_ncomp_real[dst_dat_index];
              auto dst_ptrs = k_dst_map_ptrs.d_ptr_real[dst_dat_index];
              for (int cx = 0; cx < ncomp; cx++) {
                dst_ptrs[cell][cx][dst_layer] = k_zero_value_real;
              }
            }
            for (std::size_t dx = 0; dx < k_num_int_zero; dx++) {
              const int dst_dat_index = k_dst_dat_zero_index_int[dx];
              const int ncomp = k_dst_map_ptrs.d_ncomp_int[dst_dat_index];
              auto dst_ptrs = k_dst_map_ptrs.d_ptr_int[dst_dat_index];
              for (int cx = 0; cx < ncomp; cx++) {
                dst_ptrs[cell][cx][dst_layer] = k_zero_value_int;
              }
            }
          }
        }));
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

  INT total_added = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const INT added = this->h_npart_cell.ptr[cellx];
    total_added += added;
  }
  NESOASSERT(this->get_npart_local() == total_added,
             "Interals are not self consistent.");
  NESOASSERT(total_to_add == particle_group->get_npart_local(),
             "Source ParticleGroup is not self consistent.");

  pr0.end();
  this->sycl_target->profile_map.add_region(pr0);
}

void ParticleGroup::add_particles_local(
    std::shared_ptr<ParticleSubGroup> particle_sub_group) {

  if (particle_sub_group->is_entire_particle_group()) {
    this->add_particles_local(particle_sub_group->particle_group);
  } else {
    const INT npart = particle_sub_group->get_npart_local();

    auto d_buffer = get_resource<BufferDevice<INT>,
                                 ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);

    if (npart > 0) {
      d_buffer->realloc_no_copy(3 * npart);

      auto k_cells = d_buffer->ptr;
      auto k_layers_src = k_cells + npart;
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
          cgh.parallel_for<>(sycl::range<1>(static_cast<size_t>(k_npart)),
                             [=](sycl::id<1> idx) {
                               const INT cell = k_cells[idx];
                               const INT src_layer = k_layers_src[idx];
                               const INT dst_layer = k_layers_dst[idx];
                               for (int nx = 0; nx < k_ncomp; nx++) {
                                 dst_ptr[cell][nx][dst_layer] =
                                     src_ptr[cell][nx][src_layer];
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
    }
    this->check_dats_and_group_agree();
    this->invalidate_group_version();

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_buffer);
  }
}

void ParticleGroup::add_particles_local(ParticleSet &particle_data) {
  ProfileRegion pr0("ParticleGroup", "add_particles_local_particle_set");

  this->invalidate_group_version();
  const std::size_t npart_new = static_cast<std::size_t>(particle_data.npart);
  if (npart_new == 0) {
    return;
  }

  EventStack es;

  auto lambda_test_data = [&](auto m) -> std::size_t {
    std::size_t total_ncomp = 0;
    for (auto &[sym, dat] : m) {
      if (particle_data.contains(sym)) {
        const std::size_t ncomp_correct = static_cast<std::size_t>(dat->ncomp);
        const std::size_t ncomp_to_test =
            particle_data.get(sym).size() / npart_new;
        NESOASSERT(ncomp_correct == ncomp_to_test,
                   "Ncomp of particle_data for sym " + sym.name +
                       " does not match ParticleGroup.");
        total_ncomp += ncomp_correct;
      }
    }
    return total_ncomp;
  };

  const std::size_t ncomp_to_add_real =
      lambda_test_data(this->particle_dats_real);
  const std::size_t ncomp_to_add_int =
      lambda_test_data(this->particle_dats_int);

  auto d_new_real = std::make_shared<BufferDevice<REAL>>(
      this->sycl_target, ncomp_to_add_real * npart_new);
  auto d_new_int = std::make_shared<BufferDevice<INT>>(
      this->sycl_target, ncomp_to_add_int * npart_new);

  std::vector<int> dst_dat_index_real;
  std::vector<int> dst_dat_zero_index_real;
  std::vector<REAL *> dst_dat_ptr_real;
  REAL *d_new_real_ptr = d_new_real->ptr;
  int index = 0;
  for (auto &[sym, dat] : this->particle_dats_real) {
    if (particle_data.contains(sym)) {
      const std::size_t ncomp = static_cast<std::size_t>(dat->ncomp);
      const std::size_t nelements = ncomp * npart_new;
      es.push(this->sycl_target->queue.memcpy(d_new_real_ptr,
                                              particle_data.get_ptr(sym, 0, 0),
                                              nelements * sizeof(REAL)));
      dst_dat_index_real.push_back(index);
      dst_dat_ptr_real.push_back(d_new_real_ptr);
      d_new_real_ptr += nelements;
    } else {
      dst_dat_zero_index_real.push_back(index);
    }
    index++;
  }
  std::vector<int> dst_dat_index_int;
  std::vector<int> dst_dat_zero_index_int;
  std::vector<INT *> dst_dat_ptr_int;
  INT *d_new_int_ptr = d_new_int->ptr;
  index = 0;
  for (auto &[sym, dat] : this->particle_dats_int) {
    if (particle_data.contains(sym)) {
      const std::size_t ncomp = static_cast<std::size_t>(dat->ncomp);
      const std::size_t nelements = ncomp * npart_new;
      es.push(this->sycl_target->queue.memcpy(d_new_int_ptr,
                                              particle_data.get_ptr(sym, 0, 0),
                                              nelements * sizeof(INT)));
      dst_dat_index_int.push_back(index);
      dst_dat_ptr_int.push_back(d_new_int_ptr);
      d_new_int_ptr += nelements;
    } else {
      dst_dat_zero_index_int.push_back(index);
    }
    index++;
  }

  // loop over the cells of the new particles and allocate more space in the
  // dats
  NESOASSERT(particle_data.contains(*this->cell_id_sym),
             "No cell ids found in ParticleSet.");
  auto cellids = particle_data.get(*this->cell_id_sym);
  std::vector<INT> layers(npart_new);
  for (std::size_t px = 0; px < npart_new; px++) {
    auto cellindex = cellids[px];
    NESOASSERT((cellindex >= 0) && (cellindex < this->ncell),
               "Bad particle cellid)");

    layers[px] = this->h_npart_cell.ptr[cellindex]++;
  }

  auto e0 = buffer_memcpy(this->d_npart_cell, this->h_npart_cell);
  auto d_cells =
      std::make_shared<BufferDevice<INT>>(this->sycl_target, cellids.size());
  auto d_layers =
      std::make_shared<BufferDevice<INT>>(this->sycl_target, layers.size());
  es.push(d_cells->set_async(cellids));
  es.push(d_layers->set_async(layers));
  const auto k_cells = d_cells->ptr;
  const auto k_layers = d_layers->ptr;

  e0.wait_and_throw();
  this->recompute_npart_cell_es();
  this->realloc_all_dats();
  auto k_dat_info = this->particle_group_pointer_map->get();

  auto d_dst_dat_index_real = std::make_shared<BufferDevice<int>>(
      this->sycl_target, dst_dat_index_real);
  auto d_dst_dat_index_int =
      std::make_shared<BufferDevice<int>>(this->sycl_target, dst_dat_index_int);
  auto d_dst_dat_zero_index_real = std::make_shared<BufferDevice<int>>(
      this->sycl_target, dst_dat_zero_index_real);
  auto d_dst_dat_zero_index_int = std::make_shared<BufferDevice<int>>(
      this->sycl_target, dst_dat_zero_index_int);
  auto d_dst_dat_ptr_real = std::make_shared<BufferDevice<REAL *>>(
      this->sycl_target, dst_dat_ptr_real);
  auto d_dst_dat_ptr_int =
      std::make_shared<BufferDevice<INT *>>(this->sycl_target, dst_dat_ptr_int);
  auto k_dst_dat_index_real = d_dst_dat_index_real->ptr;
  auto k_dst_dat_index_int = d_dst_dat_index_int->ptr;
  auto k_dst_dat_zero_index_real = d_dst_dat_zero_index_real->ptr;
  auto k_dst_dat_zero_index_int = d_dst_dat_zero_index_int->ptr;
  auto k_dst_dat_ptr_real = d_dst_dat_ptr_real->ptr;
  auto k_dst_dat_ptr_int = d_dst_dat_ptr_int->ptr;

  const REAL k_zero_value_real = this->zero_value_real;
  const INT k_zero_value_int = this->zero_value_int;
  es.wait();
  if (dst_dat_index_real.size()) {
    es.push(this->sycl_target->queue.parallel_for<>(
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(dst_dat_index_real.size(), npart_new)),
        [=](sycl::id<2> idx) {
          const int dx = idx[0];
          const int dat_index = k_dst_dat_index_real[dx];
          const int new_particle_index = idx[1];
          const INT cell = k_cells[new_particle_index];
          const INT layer = k_layers[new_particle_index];
          const int ncomp = k_dat_info.d_ncomp_real[dat_index];
          const REAL *data_src = k_dst_dat_ptr_real[dx];
          for (int cx = 0; cx < ncomp; cx++) {
            k_dat_info.d_ptr_real[dat_index][cell][cx][layer] =
                data_src[cx * npart_new + new_particle_index];
          }
        }));
  }
  if (dst_dat_index_int.size()) {
    es.push(this->sycl_target->queue.parallel_for<>(
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(dst_dat_index_int.size(), npart_new)),
        [=](sycl::id<2> idx) {
          const int dx = idx[0];
          const int dat_index = k_dst_dat_index_int[dx];
          const int new_particle_index = idx[1];
          const INT cell = k_cells[new_particle_index];
          const INT layer = k_layers[new_particle_index];
          const int ncomp = k_dat_info.d_ncomp_int[dat_index];
          const INT *data_src = k_dst_dat_ptr_int[dx];
          for (int cx = 0; cx < ncomp; cx++) {
            k_dat_info.d_ptr_int[dat_index][cell][cx][layer] =
                data_src[cx * npart_new + new_particle_index];
          }
        }));
  }
  if (dst_dat_zero_index_real.size()) {
    es.push(this->sycl_target->queue.parallel_for<>(
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(dst_dat_zero_index_real.size(), npart_new)),
        [=](sycl::id<2> idx) {
          const int dx = idx[0];
          const int dat_index = k_dst_dat_zero_index_real[dx];
          const int new_particle_index = idx[1];
          const INT cell = k_cells[new_particle_index];
          const INT layer = k_layers[new_particle_index];
          const int ncomp = k_dat_info.d_ncomp_real[dat_index];
          for (int cx = 0; cx < ncomp; cx++) {
            k_dat_info.d_ptr_real[dat_index][cell][cx][layer] =
                k_zero_value_real;
          }
        }));
  }
  if (dst_dat_zero_index_int.size()) {
    es.push(this->sycl_target->queue.parallel_for<>(
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(dst_dat_zero_index_int.size(), npart_new)),
        [=](sycl::id<2> idx) {
          const int dx = idx[0];
          const int dat_index = k_dst_dat_zero_index_int[dx];
          const int new_particle_index = idx[1];
          const INT cell = k_cells[new_particle_index];
          const INT layer = k_layers[new_particle_index];
          const int ncomp = k_dat_info.d_ncomp_int[dat_index];
          for (int cx = 0; cx < ncomp; cx++) {
            k_dat_info.d_ptr_int[dat_index][cell][cx][layer] = k_zero_value_int;
          }
        }));
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
  pr0.end();
  this->sycl_target->profile_map.add_region(pr0);
}

void ParticleGroup::add_particles_local(ParticleSetSharedPtr particle_data) {
  add_particles_local(*particle_data);
}

void ParticleGroup::remove_particles(const int npart,
                                     const std::vector<INT> &cells,
                                     const std::vector<INT> &layers) {

  auto r0 = this->sycl_target->profile_map.start_region("ParticleGroup",
                                                        "remove_particles_0");
  if (npart > 0) {
    auto d_buffer = get_resource<BufferDevice<INT>,
                                 ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);
    d_buffer->realloc_no_copy(2 * npart);

    auto k_cells = d_buffer->ptr;
    auto k_layers = k_cells + npart;

    NESOASSERT(cells.size() >= static_cast<std::size_t>(npart),
               "Bad cells length compared to npart");
    NESOASSERT(layers.size() >= static_cast<std::size_t>(npart),
               "Bad layers length compared to npart");

    const std::size_t num_bytes = static_cast<std::size_t>(npart) * sizeof(INT);
    auto event_cells =
        this->sycl_target->queue.memcpy(k_cells, cells.data(), num_bytes);
    auto event_layers =
        this->sycl_target->queue.memcpy(k_layers, layers.data(), num_bytes);

    event_cells.wait_and_throw();
    event_layers.wait_and_throw();

    this->remove_particles(npart, k_cells, k_layers);

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_buffer);
  }
  this->sycl_target->profile_map.end_region(r0);
}

void ParticleGroup::remove_particles(
    std::shared_ptr<ParticleSubGroup> particle_sub_group) {
  auto r0 = this->sycl_target->profile_map.start_region("ParticleGroup",
                                                        "remove_particles_1");
  NESOASSERT(particle_sub_group->particle_group.get() == this,
             "remove_particles called with a ParticleSubGroup which is not a "
             "sub group of the ParticleGroup");

  if (particle_sub_group->is_entire_particle_group()) {
    this->clear();
  } else {
    auto d_buffer = get_resource<BufferDevice<INT>,
                                 ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);

    const auto npart = particle_sub_group->get_npart_local();
    d_buffer->realloc_no_copy(2 * npart);

    auto k_cells = d_buffer->ptr;
    auto k_layers = k_cells + npart;
    particle_sub_group->get_cells_layers(k_cells, k_layers);
    this->remove_particles(npart, k_cells, k_layers);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_buffer);
  }
  this->sycl_target->profile_map.end_region(r0);
}

void ParticleGroup::global_move() {
  this->global_move_ctx.move();
  this->set_npart_cell_from_dat();
  this->invalidate_group_version();
}

void ParticleGroup::local_move() {
  this->local_move_ctx->move();
  this->set_npart_cell_from_dat();
  this->invalidate_group_version();
}

void ParticleGroup::hybrid_move() {
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

size_t ParticleGroup::particle_size() {
  size_t s = 0;
  for (auto &dat : this->particle_dats_real) {
    s += dat.second->cell_dat.row_size();
  }
  for (auto &dat : this->particle_dats_int) {
    s += dat.second->cell_dat.row_size();
  }
  return s;
};

void ParticleGroup::cell_move() {
  this->domain->local_mapper->map_cells(*this);
  this->cell_move_ctx.move();
  this->set_npart_cell_from_dat();
  this->invalidate_group_version();
};

void ParticleGroup::remove_particle_dat(Sym<REAL> sym) {
  NESOASSERT(this->particle_dats_real.count(sym) == 1,
             "ParticleDat not found.");

  NESOASSERT(sym.name != this->position_dat->sym.name,
             "The positions dat cannot be removed.");

  this->remove_particle_dat_common(this->particle_dats_real.at(sym));
  this->particle_dats_real.erase(sym);
}

void ParticleGroup::remove_particle_dat(Sym<INT> sym) {
  NESOASSERT(this->particle_dats_int.count(sym) == 1, "ParticleDat not found.");

  NESOASSERT(sym.name != "NESO_MPI_RANK",
             "The MPI rank dat cannot be removed.");
  NESOASSERT(sym.name != this->cell_id_sym->name,
             "The cell id dat cannot be removed.");

  this->remove_particle_dat_common(this->particle_dats_int.at(sym));
  this->particle_dats_int.erase(sym);
}

ParticleSetSharedPtr ParticleGroup::get_particles(std::vector<INT> &cells,
                                                  std::vector<INT> &layers) {

  auto r0 = this->sycl_target->profile_map.start_region("ParticleGroup",
                                                        "get_particles_1");

  NESOASSERT(cells.size() == layers.size(),
             "Cells and layers vectors have different sizes.");
  const int num_particles = cells.size();

  auto d_buffer =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_buffer->realloc_no_copy(2 * num_particles);

  INT *d_cells = d_buffer->ptr;
  INT *d_layers = d_cells + num_particles;

  auto e0 = this->sycl_target->queue.memcpy(d_cells, cells.data(),
                                            num_particles * sizeof(INT));
  auto e1 = this->sycl_target->queue.memcpy(d_layers, layers.data(),
                                            num_particles * sizeof(INT));

  const INT num_cells = this->domain->mesh->get_cell_count();
  for (int px = 0; px < num_particles; px++) {
    const INT cell = cells.at(px);
    const INT layer = layers.at(px);
    NESOASSERT((cell > -1) && (cell < num_cells), "Cell index not in range.");
    NESOASSERT((layer > -1) && (layer < this->get_npart_cell(cell)),
               "Layer index not in range.");
  }

  e0.wait_and_throw();
  e1.wait_and_throw();

  auto ps = this->get_particles(num_particles, d_cells, d_layers);

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_buffer);

  this->sycl_target->profile_map.end_region(r0);
  return ps;
}

} // namespace NESO::Particles
