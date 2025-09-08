#include <neso_particles/common_impl.hpp>
#include <neso_particles/particle_io.hpp>

#ifdef NESO_PARTICLES_HDF5

namespace NESO::Particles {

template void
H5Part::write_inner<ParticleGroup>(std::shared_ptr<ParticleGroup> group);
template void
H5Part::write_inner<ParticleSubGroup>(std::shared_ptr<ParticleSubGroup> group);

template void H5Part::write_dat_2d<ParticleGroup, REAL>(
    std::shared_ptr<ParticleGroup> parent, const std::int64_t npart_total,
    const std::int64_t npart_local, const std::int64_t offset,
    ParticleDatSharedPtr<REAL> dat, hid_t group_step, hid_t dxpl);
template void H5Part::write_dat_2d<ParticleGroup, INT>(
    std::shared_ptr<ParticleGroup> parent, const std::int64_t npart_total,
    const std::int64_t npart_local, const std::int64_t offset,
    ParticleDatSharedPtr<INT> dat, hid_t group_step, hid_t dxpl);
template void H5Part::write_dat_2d<ParticleSubGroup, REAL>(
    std::shared_ptr<ParticleSubGroup> parent, const std::int64_t npart_total,
    const std::int64_t npart_local, const std::int64_t offset,
    ParticleDatSharedPtr<REAL> dat, hid_t group_step, hid_t dxpl);
template void H5Part::write_dat_2d<ParticleSubGroup, INT>(
    std::shared_ptr<ParticleSubGroup> parent, const std::int64_t npart_total,
    const std::int64_t npart_local, const std::int64_t offset,
    ParticleDatSharedPtr<INT> dat, hid_t group_step, hid_t dxpl);
template void H5Part::write_dat_column_wise<ParticleGroup, REAL>(
    std::shared_ptr<ParticleGroup> parent, const std::int64_t npart_local,
    ParticleDatSharedPtr<REAL> dat, hid_t dxpl, hid_t group_step,
    hid_t memspace, hid_t filespace, bool is_position);
template void H5Part::write_dat_column_wise<ParticleGroup, INT>(
    std::shared_ptr<ParticleGroup> parent, const std::int64_t npart_local,
    ParticleDatSharedPtr<INT> dat, hid_t dxpl, hid_t group_step, hid_t memspace,
    hid_t filespace, bool is_position);

template void H5Part::write_dat_column_wise<ParticleSubGroup, REAL>(
    std::shared_ptr<ParticleSubGroup> parent, const std::int64_t npart_local,
    ParticleDatSharedPtr<REAL> dat, hid_t dxpl, hid_t group_step,
    hid_t memspace, hid_t filespace, bool is_position);
template void H5Part::write_dat_column_wise<ParticleSubGroup, INT>(
    std::shared_ptr<ParticleSubGroup> parent, const std::int64_t npart_local,
    ParticleDatSharedPtr<INT> dat, hid_t dxpl, hid_t group_step, hid_t memspace,
    hid_t filespace, bool is_position);

void H5Part::write(INT step_in) {
  NESOASSERT(this->particle_group != nullptr,
             "There is no particle group to write from, maybe this H5Part "
             "instance is in read-only mode?");

  // open the file for writing if required.
  this->open_read_write();

  if (step_in >= 0) {
    this->step = step_in;
  }

  if (this->particle_sub_group != nullptr) {
    this->write_inner(this->particle_sub_group);
  } else {
    this->write_inner(this->particle_group);
  }

  this->step++;
}

ParticleSetSharedPtr H5Part::read(ParticleSpec &particle_spec, INT step,
                                  const bool use_xyz_positions) {
  this->open_read();

  std::string step_name = "Step#";
  step_name += std::to_string(step);

  hid_t group_step;
  H5CHK(group_step = H5Gopen(this->file_id, step_name.c_str(), H5P_DEFAULT));
  INT npart_total;

  {
    hid_t dataset;
    H5CHK(dataset = H5Dopen(group_step, "x", H5P_DEFAULT));
    hid_t filespace;
    H5CHK(filespace = H5Dget_space(dataset));

    hsize_t d_rank = H5Sget_simple_extent_ndims(filespace);
    NESOASSERT(d_rank == 1, "Expected x to be a column.");

    hsize_t dims[1];
    H5CHK(H5Sget_simple_extent_dims(filespace, dims, NULL));
    H5CHK(H5Dclose(dataset));
    npart_total = dims[0];
  }

  INT offset_start = -1;
  INT offset_end = -1;
  get_decomp_1d(static_cast<INT>(comm_pair.size_parent), npart_total,
                static_cast<INT>(comm_pair.rank_parent), &offset_start,
                &offset_end);
  const INT npart_local = offset_end - offset_start;

  auto particle_set = std::make_shared<ParticleSet>(npart_local, particle_spec);

  if (npart_total > 0) {

    // Create a memspace and filespace for the columnwise reads.
    hsize_t dims_memspace[1] = {static_cast<hsize_t>(npart_local)};
    hid_t memspace = H5Screate_simple(1, dims_memspace, NULL);

    hsize_t dims_filespace[1] = {static_cast<hsize_t>(npart_total)};
    hid_t filespace = H5Screate_simple(1, dims_filespace, NULL);

    hsize_t slab_offsets[1] = {static_cast<hsize_t>(offset_start)};
    hsize_t slab_counts[1] = {static_cast<hsize_t>(npart_local)};

    // select the hyperslab for the columnwite reads.
    H5CHK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, slab_offsets, NULL,
                              slab_counts, NULL));
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5CHK(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));

    auto lambda_read_data = [&](const std::string name, const auto sym,
                                auto *h_ptr) {
      hid_t dset = H5Dopen(group_step, name.c_str(), H5P_DEFAULT);
      H5CHK(H5Dread(dset, memtypeid(sym), memspace, filespace, dxpl, h_ptr));
      H5CHK(H5Dclose(dset));
    };

    for (auto &px : particle_spec.properties_real) {
      if (px.positions && use_xyz_positions) {
        for (int cx = 0; cx < px.ncomp; cx++) {
          lambda_read_data(position_labels[cx], px.sym,
                           particle_set->get_ptr(px.sym, 0, cx));
        }
      } else {
        for (int cx = 0; cx < px.ncomp; cx++) {
          lambda_read_data(px.sym.name + "_" + std::to_string(cx), px.sym,
                           particle_set->get_ptr(px.sym, 0, cx));
        }
      }
    }
    for (auto &px : particle_spec.properties_int) {
      for (int cx = 0; cx < px.ncomp; cx++) {
        lambda_read_data(px.sym.name + "_" + std::to_string(cx), px.sym,
                         particle_set->get_ptr(px.sym, 0, cx));
      }
    }

    H5CHK(H5Pclose(dxpl));
    H5CHK(H5Sclose(filespace));
    H5CHK(H5Sclose(memspace));
  }
  H5CHK(H5Gclose(group_step));
  return particle_set;
}

} // namespace NESO::Particles

#endif
