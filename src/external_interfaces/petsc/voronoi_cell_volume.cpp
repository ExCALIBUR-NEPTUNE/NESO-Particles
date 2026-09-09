#ifdef NESO_PARTICLES_PETSC

#include <neso_particles/external_interfaces/petsc/voronoi_cell_volume.hpp>

namespace NESO::Particles::PetscInterface {

void estimate_voronoi_cell_volume(DMPlexInterfaceSharedPtr mesh,
                                  SubdivideCellsVoronoiSharedPtr voronoi_cells,
                                  std::size_t &num_samples, REAL &stol,
                                  std::size_t max_num_samples,
                                  NDLocalArraySharedPtr<REAL, 2> &volumes,
                                  std::mt19937 *rng_in,
                                  const int default_block_size) {

  std::mt19937 rng;
  if (rng_in == nullptr) {
    rng = std::mt19937(std::random_device{}());
    rng_in = &rng;
  }

  auto sycl_target = voronoi_cells->sycl_target;
  auto r0 = sycl_target->profile_map.start_region(
      "estimate_voronoi_cell_volume", "all");

  const int ndim = mesh->get_ndim();
  auto mapper = std::make_shared<DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const int cell_count = mesh->get_cell_count();
  const INT max_num_voronoi_cells =
      std::max(1, voronoi_cells->points->get_nrow_max());

  const std::size_t block_size = default_block_size * max_num_voronoi_cells;
  const std::size_t min_num_samples_per_voronoi_cell = num_samples;
  const std::size_t max_num_samples_per_voronoi_cell = max_num_samples;

  const std::size_t min_num_samples = get_next_multiple(
      min_num_samples_per_voronoi_cell * max_num_voronoi_cells, block_size);
  max_num_samples = max_num_samples_per_voronoi_cell * max_num_voronoi_cells;

  ParticleSet initial_distribution(cell_count * block_size, particle_spec);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (std::size_t layerx = 0; layerx < block_size; layerx++) {
      initial_distribution[Sym<INT>("CELL_ID")][cellx * block_size + layerx]
                          [0] = cellx;
    }
  }
  A->add_particles_local(initial_distribution);

  const int num_particles = block_size * cell_count;
  std::vector<std::vector<double>> positions(ndim);
  for (int dx = 0; dx < ndim; dx++) {
    positions.at(dx).resize(num_particles);
  }
  std::vector<int> cells(num_particles);

  auto d_positions = get_resource<BufferDevice<REAL>,
                                  ResourceStackInterfaceBufferDevice<REAL>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);
  d_positions->realloc_no_copy(num_particles * ndim);
  REAL *RESTRICT k_positions = d_positions->ptr;

  auto d_counts =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_counts->realloc_no_copy(cell_count * max_num_voronoi_cells);
  INT *RESTRICT k_counts = d_counts->ptr;
  sycl_target->queue.fill<INT>(k_counts, 0, cell_count * max_num_voronoi_cells)
      .wait_and_throw();

  auto d_num_voronoi_cells =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_num_voronoi_cells->realloc_no_copy(cell_count);
  INT *RESTRICT k_num_voronoi_cells = d_num_voronoi_cells->ptr;
  sycl_target->queue
      .memcpy(k_num_voronoi_cells, voronoi_cells->points->nrow.data(),
              cell_count * sizeof(INT))
      .wait_and_throw();

  auto d_volumes_a = get_resource<BufferDevice<REAL>,
                                  ResourceStackInterfaceBufferDevice<REAL>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);
  d_volumes_a->realloc_no_copy(max_num_voronoi_cells * cell_count);
  REAL *k_volumes_a = d_volumes_a->ptr;
  auto d_volumes_b = get_resource<BufferDevice<REAL>,
                                  ResourceStackInterfaceBufferDevice<REAL>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);
  d_volumes_b->realloc_no_copy(max_num_voronoi_cells * cell_count);
  REAL *k_volumes_b = d_volumes_b->ptr;

  sycl_target->queue
      .fill<REAL>(k_volumes_a, -1.0, cell_count * max_num_voronoi_cells)
      .wait_and_throw();
  sycl_target->queue
      .fill<REAL>(k_volumes_b, -1.0, cell_count * max_num_voronoi_cells)
      .wait_and_throw();

  auto d_stol = get_resource<BufferDevice<REAL>,
                             ResourceStackInterfaceBufferDevice<REAL>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);
  d_stol->realloc_no_copy(1);
  REAL *k_stol = d_stol->ptr;

  EventStack event_stack;

  const REAL stol_in = stol;
  REAL cstol = stol + 1.0;
  bool converged = false;

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  auto iteration_set_cells = sycl_target->device_limits.validate_nd_range(
      sycl::nd_range<2>(sycl::range<2>(cell_count, local_size),
                        sycl::range<2>(1, local_size)));

  num_samples = 0;
  for (std::size_t bx = 0; (bx < max_num_samples) && (!converged);
       bx += block_size) {

    uniform_within_dmplex_cells(mesh, block_size, positions, cells, rng_in,
                                max_num_samples);

    for (int dx = 0; dx < ndim; dx++) {
      event_stack.push(sycl_target->queue.memcpy(
          k_positions + dx * num_particles, positions.at(dx).data(),
          num_particles * sizeof(REAL)));
    }
    event_stack.wait();

    particle_loop(
        "estimate_voronoi_cell_volume::position_copy", A,
        [=](auto INDEX, auto P) {
          const auto index = INDEX.get_local_linear_index();
          for (int dx = 0; dx < ndim; dx++) {
            P.at(dx) = k_positions[dx * num_particles + index];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("P")))
        ->execute();

    voronoi_cells->map(A, Sym<INT>("CELL_ID"), 0);

    particle_loop(
        "estimate_voronoi_cell_volume::increment_counter", A,
        [=](auto INDEX, auto CELL_ID) {
          const auto cell = INDEX.cell;
          const INT voronoi_cell = CELL_ID.at(0);
          if ((0 <= voronoi_cell) && (voronoi_cell < max_num_voronoi_cells)) {
            atomic_fetch_add<INT>(
                &k_counts[max_num_voronoi_cells * cell + voronoi_cell], 1);
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(Sym<INT>("CELL_ID")))
        ->execute();

    sycl_target->queue
        .parallel_for(
            iteration_set_cells,
            [=](sycl::nd_item<2> idx) {
              const std::size_t mesh_cell = idx.get_global_id(0);
              const INT local_id = idx.get_global_id(1);
              const INT num_voronoi_cells = k_num_voronoi_cells[mesh_cell];

              INT contrib = 0;
              for (INT vcellx = local_id; vcellx < num_voronoi_cells;
                   vcellx += local_size) {
                contrib += k_counts[max_num_voronoi_cells * mesh_cell + vcellx];
              }
              const INT total = sycl::reduce_over_group(
                  idx.get_group(), contrib, sycl::plus<INT>{});
              const REAL total_real = total;

              for (INT vcellx = local_id; vcellx < num_voronoi_cells;
                   vcellx += local_size) {
                const REAL contrib_real =
                    k_counts[max_num_voronoi_cells * mesh_cell + vcellx];
                const REAL ratio = contrib_real / total_real;
                k_volumes_a[max_num_voronoi_cells * mesh_cell + vcellx] = ratio;
              }
            })
        .wait_and_throw();

    // Have we passed the first set of samples and hence can compute an stol?
    if (bx >= block_size) {

      auto e_reset = sycl_target->queue.fill<REAL>(k_stol, 0.0, 1);
      auto e0 = sycl_target->queue.parallel_for(
          iteration_set_cells, e_reset, [=](sycl::nd_item<2> idx) {
            const std::size_t mesh_cell = idx.get_global_id(0);
            const INT local_id = idx.get_global_id(1);
            const INT num_voronoi_cells = k_num_voronoi_cells[mesh_cell];

            REAL stol_contrib = 0.0;
            for (INT vcellx = local_id; vcellx < num_voronoi_cells;
                 vcellx += local_size) {
              const REAL ratio_a =
                  k_volumes_a[max_num_voronoi_cells * mesh_cell + vcellx];
              const REAL ratio_b =
                  k_volumes_b[max_num_voronoi_cells * mesh_cell + vcellx];

              const REAL stol_ab = Kernel::relative_error(ratio_a, ratio_b);
              stol_contrib = sycl::max(stol_contrib, stol_ab);
            }

            const REAL stol_reduced = sycl::reduce_over_group(
                idx.get_group(), stol_contrib, sycl::maximum<REAL>{});

            if (idx.get_group().leader()) {
              atomic_fetch_max<REAL>(k_stol, stol_reduced);
            }
          });

      REAL last_stol = stol_in + 1.0;
      sycl_target->queue.memcpy(&last_stol, k_stol, sizeof(REAL), e0)
          .wait_and_throw();
      converged = (last_stol < stol_in) && (num_samples >= min_num_samples);
      stol = last_stol;
    }

    {
      REAL *tmp = k_volumes_a;
      k_volumes_a = k_volumes_b;
      k_volumes_b = tmp;
    }
    num_samples += block_size;
  }

  auto h_volumes =
      get_resource<BufferHost<REAL>, ResourceStackInterfaceBufferHost<REAL>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferHost<REAL>{},
          sycl_target);
  h_volumes->realloc_no_copy(cell_count);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const REAL mesh_cell_volume = mesh->dmh->get_cell_volume(cellx);
    h_volumes->ptr[cellx] = mesh_cell_volume;
  }

  auto e0 = sycl_target->queue.memcpy(k_volumes_a, h_volumes->ptr,
                                      cell_count * sizeof(REAL));

  if (volumes == nullptr) {
    volumes = std::make_shared<NDLocalArray<REAL, 2>>(sycl_target, cell_count,
                                                      max_num_voronoi_cells);
  } else if ((volumes->index.shape[0] != cell_count) ||
             (volumes->index.shape[1] != max_num_voronoi_cells)) {
    volumes = std::make_shared<NDLocalArray<REAL, 2>>(sycl_target, cell_count,
                                                      max_num_voronoi_cells);
  }

  volumes->fill(-1.0);
  e0.wait_and_throw();

  REAL *RESTRICT k_volumes = volumes->ptr();
  sycl_target->queue
      .parallel_for(
          iteration_set_cells,
          [=](sycl::nd_item<2> idx) {
            const std::size_t mesh_cell = idx.get_global_id(0);
            const INT local_id = idx.get_global_id(1);
            const INT num_voronoi_cells = k_num_voronoi_cells[mesh_cell];
            const REAL mesh_cell_volume = k_volumes_a[mesh_cell];

            for (INT vcellx = local_id; vcellx < num_voronoi_cells;
                 vcellx += local_size) {
              const REAL ratio_b =
                  k_volumes_b[max_num_voronoi_cells * mesh_cell + vcellx];
              const REAL vcell_volume = mesh_cell_volume * ratio_b;
              k_volumes[mesh_cell * max_num_voronoi_cells + vcellx] =
                  vcell_volume;
            }
          })
      .wait_and_throw();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferHost<REAL>{}, h_volumes);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<REAL>{}, d_stol);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<REAL>{}, d_volumes_b);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<REAL>{}, d_volumes_a);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_num_voronoi_cells);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_counts);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<REAL>{}, d_positions);
  sycl_target->profile_map.end_region(r0);
}

} // namespace NESO::Particles::PetscInterface

#endif
