#include <neso_particles/containers/particle_set_device.hpp>

namespace NESO::Particles {

ParticleSetDeviceSpec::ParticleSetDeviceSpec(ParticleSpec &particle_spec)
    : particle_spec(particle_spec) {

  this->num_properties_real = particle_spec.properties_real.size();
  this->num_properties_int = particle_spec.properties_int.size();
  this->components_real = std::vector<int>(num_properties_real);
  this->components_int = std::vector<int>(num_properties_int);
  this->syms_real = std::vector<Sym<REAL>>(num_properties_real);
  this->syms_int = std::vector<Sym<INT>>(num_properties_int);

  for (int px = 0; px < num_properties_real; px++) {
    this->components_real.at(px) = particle_spec.properties_real.at(px).ncomp;
    const auto sym = particle_spec.properties_real.at(px).sym;
    this->syms_real.at(px) = sym;
    this->map_sym_index_real[sym] = px;
  }
  for (int px = 0; px < num_properties_int; px++) {
    this->components_int.at(px) = particle_spec.properties_int.at(px).ncomp;
    const auto sym = particle_spec.properties_int.at(px).sym;
    this->syms_int.at(px) = sym;
    this->map_sym_index_int[sym] = px;
  }
  this->num_components_real = std::accumulate(this->components_real.begin(),
                                              this->components_real.end(), 0);
  this->num_components_int = std::accumulate(this->components_int.begin(),
                                             this->components_int.end(), 0);
}

void ParticleSetDeviceSpec::set_default_value(Sym<INT> sym, const int component,
                                              const INT value) {
  this->default_values_int[{sym, component}] = value;
}
void ParticleSetDeviceSpec::set_default_value(Sym<REAL> sym,
                                              const int component,
                                              const REAL value) {
  this->default_values_real[{sym, component}] = value;
}

int ParticleSetDeviceSpec::get_sym_index(const Sym<REAL> sym) const {
  auto it = this->map_sym_index_real.find(sym);
  if (it == this->map_sym_index_real.end()) {
    return -1;
  } else {
    return it->second;
  }
}

int ParticleSetDeviceSpec::get_sym_index(const Sym<INT> sym) const {
  auto it = this->map_sym_index_int.find(sym);
  if (it == this->map_sym_index_int.end()) {
    return -1;
  } else {
    return it->second;
  }
}

int ParticleSetDeviceSpec::get_num_components(const Sym<REAL> sym) {
  const int index = this->get_sym_index(sym);
  if (index < 0) {
    return 0;
  } else {
    return this->components_real.at(index);
  }
}

int ParticleSetDeviceSpec::get_num_components(const Sym<INT> sym) {
  const int index = this->get_sym_index(sym);
  if (index < 0) {
    return 0;
  } else {
    return this->components_int.at(index);
  }
}

ParticleSetDevice::ParticleSetDevice(
    SYCLTargetSharedPtr sycl_target,
    std::shared_ptr<ParticleSetDeviceSpec> spec)
    : sycl_target(sycl_target), num_particles(0), spec(spec) {

  NESOASSERT(sycl_target != nullptr, "sycl_target is nullptr.");
  this->d_data_real = std::make_shared<BufferDevice<REAL>>(sycl_target, 1);
  this->d_data_int = std::make_shared<BufferDevice<INT>>(sycl_target, 1);

  // These offsets are functions of the properties and components not the
  // number of particles

  // REAL offsets
  std::vector<int> h_offsets_real(spec->components_real.size());
  std::exclusive_scan(spec->components_real.begin(),
                      spec->components_real.end(), h_offsets_real.begin(), 0);

  this->dh_offsets_real =
      std::make_shared<BufferDeviceHost<int>>(sycl_target, h_offsets_real);
  this->dh_offsets_real->host_to_device();

  // INT offsets
  std::vector<int> h_offsets_int(spec->components_int.size());
  std::exclusive_scan(spec->components_int.begin(), spec->components_int.end(),
                      h_offsets_int.begin(), 0);

  this->dh_offsets_int =
      std::make_shared<BufferDeviceHost<int>>(sycl_target, h_offsets_int);
  this->dh_offsets_int->host_to_device();
}

ParticleSetDevice::ParticleSetDevice(SYCLTargetSharedPtr sycl_target,
                                     const int num_particles,
                                     ParticleSpec &particle_spec)
    : ParticleSetDevice(
          sycl_target, std::make_shared<ParticleSetDeviceSpec>(particle_spec)) {
  this->reset(num_particles);
}

void ParticleSetDevice::reset(const int num_particles) {
  const auto spec = this->spec.get();
  NESOASSERT(spec != nullptr, "ParticleSetDevice is not initialised.");
  NESOASSERT(num_particles >= 0,
             "A negative number of particles does not make sense.");
  this->num_particles = num_particles;
  if (num_particles > 0) {
    this->d_data_real->realloc_no_copy(num_particles *
                                       spec->num_components_real);
    this->d_data_int->realloc_no_copy(num_particles * spec->num_components_int);
    EventStack es;

    // reset the values to either 0 or the set default value
    for (int sx = 0; sx < spec->num_properties_real; sx++) {
      for (int cx = 0; cx < spec->components_real[sx]; cx++) {
        const std::pair<Sym<REAL>, int> key = {spec->syms_real.at(sx), cx};
        const REAL value = spec->default_values_real.count(key) > 0
                               ? spec->default_values_real.at(key)
                               : 0;
        const int sym_offset = this->dh_offsets_real->h_buffer.ptr[sx];
        REAL *column_start =
            this->d_data_real->ptr + (sym_offset + cx) * num_particles;
        es.push(
            this->sycl_target->queue.fill(column_start, value, num_particles));
      }
    }
    for (int sx = 0; sx < spec->num_properties_int; sx++) {
      for (int cx = 0; cx < spec->components_int[sx]; cx++) {
        const std::pair<Sym<INT>, int> key = {spec->syms_int.at(sx), cx};
        const INT value = spec->default_values_int.count(key) > 0
                              ? spec->default_values_int.at(key)
                              : 0;
        const int sym_offset = this->dh_offsets_int->h_buffer.ptr[sx];
        INT *column_start =
            this->d_data_int->ptr + (sym_offset + cx) * num_particles;
        es.push(
            this->sycl_target->queue.fill(column_start, value, num_particles));
      }
    }

    es.wait();
  }
}

ParticleSetSharedPtr ParticleSetDevice::get() {
  auto particle_set = std::make_shared<ParticleSet>(this->num_particles,
                                                    this->spec->particle_spec);
  if (this->num_particles > 0) {
    EventStack es;

    for (int sx = 0; sx < spec->num_properties_real; sx++) {
      const int num_components = this->spec->components_real[sx];
      const auto sym = this->spec->syms_real[sx];
      const int sym_offset = this->dh_offsets_real->h_buffer.ptr[sx];
      auto ptr_dst = particle_set->get_ptr(sym, 0, 0);
      const auto ptr_src =
          this->d_data_real->ptr + sym_offset * this->num_particles;
      const std::size_t num_bytes =
          this->num_particles * num_components * sizeof(REAL);
      es.push(this->sycl_target->queue.memcpy(ptr_dst, ptr_src, num_bytes));
    }
    for (int sx = 0; sx < spec->num_properties_int; sx++) {
      const int num_components = this->spec->components_int[sx];
      const auto sym = this->spec->syms_int[sx];
      const int sym_offset = this->dh_offsets_int->h_buffer.ptr[sx];
      auto ptr_dst = particle_set->get_ptr(sym, 0, 0);
      const auto ptr_src =
          this->d_data_int->ptr + sym_offset * this->num_particles;
      const std::size_t num_bytes =
          this->num_particles * num_components * sizeof(INT);
      es.push(this->sycl_target->queue.memcpy(ptr_dst, ptr_src, num_bytes));
    }

    es.wait();
  }
  return particle_set;
}

void ParticleSetDevice::set(ParticleSetSharedPtr particle_set) {
  NESOASSERT(this->num_particles == particle_set->npart,
             "Missmatch in number of particles.");
  if (this->num_particles > 0) {
    EventStack es;
    for (int sx = 0; sx < spec->num_properties_real; sx++) {
      const auto sym = this->spec->syms_real[sx];
      if (particle_set->contains(sym)) {
        const int num_components = this->spec->components_real[sx];
        NESOASSERT(num_components == particle_set->ncomp_real.at(sym),
                   "Missmatch in number of components for sym with name: " +
                       sym.name);
        const int sym_offset = this->dh_offsets_real->h_buffer.ptr[sx];
        auto ptr_src = particle_set->get_ptr(sym, 0, 0);
        const auto ptr_dst =
            this->d_data_real->ptr + sym_offset * this->num_particles;
        const std::size_t num_bytes =
            this->num_particles * num_components * sizeof(REAL);
        es.push(this->sycl_target->queue.memcpy(ptr_dst, ptr_src, num_bytes));
      }
    }
    for (int sx = 0; sx < spec->num_properties_int; sx++) {
      const auto sym = this->spec->syms_int[sx];
      if (particle_set->contains(sym)) {
        const int num_components = this->spec->components_int[sx];
        NESOASSERT(num_components == particle_set->ncomp_int.at(sym),
                   "Missmatch in number of components for sym with name: " +
                       sym.name);
        const int sym_offset = this->dh_offsets_int->h_buffer.ptr[sx];
        auto ptr_src = particle_set->get_ptr(sym, 0, 0);
        const auto ptr_dst =
            this->d_data_int->ptr + sym_offset * this->num_particles;
        const std::size_t num_bytes =
            this->num_particles * num_components * sizeof(INT);
        es.push(this->sycl_target->queue.memcpy(ptr_dst, ptr_src, num_bytes));
      }
    }

    es.wait();
  }
}

} // namespace NESO::Particles
