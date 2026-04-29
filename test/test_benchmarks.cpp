#include <gtest/gtest.h>

#include "include/test_neso_particles.hpp"
#include <iomanip>
#include <iostream>
#include <neso_particles.hpp>
#include <random>
#include <type_traits>

using namespace NESO::Particles;

static const bool benchmark_enabled =
    get_env_string("NESO_PARTICLES_ENABLE_BENCHMARK", "").size() > 0;

TEST(Benchmark, bandwidth_device_copy) {
  if (benchmark_enabled) {
    auto sycl_target =
        std::make_shared<SYCLTarget>(GPU_SELECTOR, MPI_COMM_WORLD);
    const bool root = sycl_target->comm_pair.rank_parent == 0;
    const int Ntest = 20;

    std::ofstream out_stream;
    if (root) {
      out_stream = std::ofstream("benchmark_bandwidth_device_copy.csv");
    }

    if (root) {
      auto lambda_print = [&](auto &os) {
        os << "Size (Bytes), STD Host To Device (GB/s), STD Device To Host "
              "(GB/s), SYCL Host To Device (GB/s), SYCL Device To Host (GB/s)"
           << std::endl;
      };
      lambda_print(std::cout);
      lambda_print(out_stream);
    }
    for (int px = 0; px < 31; px++) {
      const std::size_t N =
          std::pow(static_cast<std::size_t>(2), static_cast<std::size_t>(px));

      auto lambda_do_run = [&](const bool to_device, unsigned char *h_ptr,
                               unsigned char *d_ptr) -> REAL {
        MPICHK(MPI_Barrier(MPI_COMM_WORLD));
        auto t0 = profile_timestamp();
        for (int testx = 0; testx < Ntest; testx++) {
          if (to_device) {
            sycl_target->queue.memcpy(d_ptr, h_ptr, N).wait();
          } else {
            sycl_target->queue.memcpy(h_ptr, d_ptr, N).wait();
          }
        }
        MPICHK(MPI_Barrier(MPI_COMM_WORLD));
        auto t1 = profile_timestamp();
        const REAL time_elapsed = profile_elapsed(t0, t1);
        const REAL time_mean = time_elapsed / Ntest;
        const double bandwidth = N / time_mean;

        double total_bandwidth = 0.0;
        MPICHK(MPI_Reduce(&bandwidth, &total_bandwidth, 1, MPI_DOUBLE, MPI_SUM,
                          0, MPI_COMM_WORLD));
        total_bandwidth /= 1.0e9;

        return total_bandwidth;
      };

      BufferDevice<unsigned char> d_buffer(sycl_target, N);

      REAL std_host_to_device = 0.0;
      REAL std_device_to_host = 0.0;

      {
        std::vector<unsigned char> h_buffer(N);
        std::fill(h_buffer.begin(), h_buffer.end(), 1);
        std_host_to_device = lambda_do_run(true, h_buffer.data(), d_buffer.ptr);
        std_device_to_host =
            lambda_do_run(false, h_buffer.data(), d_buffer.ptr);
      }

      REAL sycl_host_to_device = 0.0;
      REAL sycl_device_to_host = 0.0;

      {
        BufferHost<unsigned char> h_buffer(sycl_target, N);
        std::fill(h_buffer.ptr, h_buffer.ptr + N, 1);
        sycl_target->queue.memcpy(h_buffer.ptr, d_buffer.ptr, N)
            .wait_and_throw();
        sycl_host_to_device = lambda_do_run(true, h_buffer.ptr, d_buffer.ptr);
        sycl_device_to_host = lambda_do_run(false, h_buffer.ptr, d_buffer.ptr);
      }

      if (root) {
        auto lambda_print = [&](auto &os) {
          os << std::setfill(' ') << std::setw(12) << N << "," << std::setw(16)
             << std::scientific << std_host_to_device << "," << std::setw(16)
             << std::scientific << std_device_to_host << "," << std::setw(16)
             << std::scientific << sycl_host_to_device << "," << std::setw(16)
             << std::scientific << sycl_device_to_host << std::endl;
        };
        lambda_print(std::cout);
        lambda_print(out_stream);
      }
    }

    std::cout << std::flush;
    if (root) {
      out_stream << std::flush;
      out_stream.close();
    }
    MPICHK(MPI_Barrier(MPI_COMM_WORLD));
    sycl_target->free();
  }
}

TEST(Benchmark, bandwidth_triad_REAL) {
  if (benchmark_enabled) {

    const int ndim = 2;
    std::vector<int> dims(ndim);
    dims[0] = 32;
    dims[1] = 32;

    auto mesh =
        std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, 1, 0);
    auto sycl_target =
        std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
    auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
    auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<REAL>("A"), 1),
                               ParticleProp(Sym<REAL>("B"), 3)};
    auto A =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
    const int cell_count = mesh->get_cell_count();

    const int comm_size = sycl_target->comm_pair.size_parent;
    const int comm_rank = sycl_target->comm_pair.rank_parent;
    const bool root = sycl_target->comm_pair.rank_parent == 0;
    const int Ntest = 400;

    std::ofstream out_stream;
    if (root) {
      out_stream = std::ofstream("benchmark_bandwidth_triad_REAL.csv");
    }

    if (root) {
      auto lambda_print = [&](auto &os) {
        os << "Size (Bytes), Bandwidth (GB/s), Time Taken (s)" << std::endl;
      };
      lambda_print(std::cout);
      lambda_print(out_stream);
    }
    for (int px = 0; px < 16; px++) {
      const std::size_t N =
          std::pow(static_cast<std::size_t>(2), static_cast<std::size_t>(px)) *
          dims[0] * dims[1];

      std::size_t start = 0;
      std::size_t end = 0;
      get_decomp_1d(comm_size, N, comm_rank, &start, &end);
      std::size_t Nlocal = end - start;

      A->clear();
      ParticleSet initial_distribution(Nlocal, particle_spec);
      for (std::size_t px = 0; px < Nlocal; px++) {
        initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
      }

      A->add_particles_local(initial_distribution);

      auto loop = particle_loop(
          A, [=](auto A, auto B) { A.at(0) = B.at(0) + B.at(1) * B.at(2); },
          Access::write(Sym<REAL>("A")), Access::read(Sym<REAL>("B")));

      MPICHK(MPI_Barrier(MPI_COMM_WORLD));
      auto t0 = profile_timestamp();

      for (int testx = 0; testx < Ntest; testx++) {
        loop->execute();
      }

      MPICHK(MPI_Barrier(MPI_COMM_WORLD));
      auto t1 = profile_timestamp();
      auto time_taken = profile_elapsed(t0, t1);

      const std::size_t num_bytes = N * 4 * sizeof(REAL);
      auto bw = Ntest * num_bytes / time_taken;
      bw /= 1e9;

      if (root) {
        auto lambda_print = [&](auto &os) {
          os << std::setfill(' ') << std::setw(12) << num_bytes << ","
             << std::setw(16) << std::scientific << bw << "," << std::setw(16)
             << std::scientific << time_taken << std::endl;
        };
        lambda_print(std::cout);
        lambda_print(out_stream);
      }
    }

    std::cout << std::flush;
    if (root) {
      out_stream << std::flush;
      out_stream.close();
    }
    MPICHK(MPI_Barrier(MPI_COMM_WORLD));
    sycl_target->free();
    mesh->free();
  }
}

TEST(Benchmark, bandwidth_triad_REAL_test) {
  if (benchmark_enabled) {

    const int ndim = 2;
    std::vector<int> dims(ndim);
    dims[0] = 32;
    dims[1] = 32;

    auto mesh =
        std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, 1, 0);
    auto sycl_target =
        std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
    auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
    auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<REAL>("A"), 1),
                               ParticleProp(Sym<REAL>("B"), 3)};
    auto A =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
    const int cell_count = mesh->get_cell_count();

    const int comm_size = sycl_target->comm_pair.size_parent;
    const int comm_rank = sycl_target->comm_pair.rank_parent;
    const bool root = sycl_target->comm_pair.rank_parent == 0;
    const int Ntest = 400;

    std::ofstream out_stream;
    if (root) {
      out_stream = std::ofstream("benchmark_bandwidth_triad_REAL.csv");
    }

    if (root) {
      auto lambda_print = [&](auto &os) {
        os << "Size (Bytes), Bandwidth (GB/s), Time Taken (s)" << std::endl;
      };
      lambda_print(std::cout);
      lambda_print(out_stream);
    }
    for (int px = 0; px < 16; px++) {
      const std::size_t N =
          std::pow(static_cast<std::size_t>(2), static_cast<std::size_t>(px)) *
          dims[0] * dims[1];

      std::size_t start = 0;
      std::size_t end = 0;
      get_decomp_1d(comm_size, N, comm_rank, &start, &end);
      std::size_t Nlocal = end - start;

      A->clear();
      ParticleSet initial_distribution(Nlocal, particle_spec);
      for (std::size_t px = 0; px < Nlocal; px++) {
        initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
      }

      A->add_particles_local(initial_distribution);

      auto loop = particle_loop(
          A, [=](auto A, auto B) { A.at(0) = B.at(0) + B.at(1) * B.at(2); },
          Access::write(Sym<REAL>("A")), Access::read(Sym<REAL>("B")));

      auto d_Aa = Access::direct_get(Access::write(A->get_dat(Sym<REAL>("A"))));
      auto d_Ba = Access::direct_get(Access::read(A->get_dat(Sym<REAL>("B"))));

      ParticleLoopImplementation::ParticleLoopBlockIterationSet
          block_iteration_set(A->get_dat(Sym<REAL>("P")));
      auto iteration_set = block_iteration_set.get_all_cells();
      EventStack event_stack;

      auto d_npart_cell = A->position_dat->d_npart_cell;

      auto lambda_foo =
          [&](REAL *RESTRICT const *RESTRICT const *RESTRICT const d_A,
              const REAL *RESTRICT const *RESTRICT const *RESTRICT d_B) {
            const auto k_npart_cell = d_npart_cell;

            const std::size_t local_size = 32;
            // sycl_target->parameters
            //     ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            //     ->value;

            sycl::nd_range<2> r(sycl::range<2>(cell_count, local_size),
                                sycl::range<2>(1, local_size));

            event_stack.push(
                sycl_target->queue.parallel_for(r, [=](sycl::nd_item<2> idx) {
                  // Get the particle cell and layer for this workitem.
                  std::size_t cell = idx.get_global_id(0);
                  const std::size_t num_layers = k_npart_cell[cell];

                  std::size_t local_id = idx.get_local_id(1);
                  std::size_t local_range = idx.get_local_range(1);

                  for (std::size_t layer = local_id; layer < num_layers;
                       layer += 32) {
                    // d_A[cell][0][layer] =
                    //     d_B[cell][0][layer] +
                    //     d_B[cell][1][layer] * d_B[cell][2][layer];

                    d_A[cell][0][layer] =
                        sycl::fma(d_B[cell][0][layer], d_B[cell][1][layer],
                                  d_B[cell][2][layer]);
                  }
                }));

            event_stack.wait();
          };

      MPICHK(MPI_Barrier(MPI_COMM_WORLD));
      auto t0 = profile_timestamp();

      // for (int testx = 0; testx < Ntest; testx++) {
      //   loop->execute();
      // }

      for (int testx = 0; testx < Ntest; testx++) {
        lambda_foo(d_Aa, d_Ba);
      }

      MPICHK(MPI_Barrier(MPI_COMM_WORLD));
      auto t1 = profile_timestamp();

      // Access::direct_restore(Access::write(A->get_dat(Sym<REAL>("A"))),
      // d_Aa); Access::direct_restore(Access::read(A->get_dat(Sym<REAL>("B"))),
      // d_Ba);

      auto time_taken = profile_elapsed(t0, t1);

      const std::size_t num_bytes = N * 4 * sizeof(REAL);
      auto bw = Ntest * num_bytes / time_taken;
      bw /= 1e9;

      if (root) {
        auto lambda_print = [&](auto &os) {
          os << std::setfill(' ') << std::setw(12) << num_bytes << ","
             << std::setw(16) << std::scientific << bw << "," << std::setw(16)
             << std::scientific << time_taken << std::endl;
        };
        lambda_print(std::cout);
        lambda_print(out_stream);
      }
    }

    std::cout << std::flush;
    if (root) {
      out_stream << std::flush;
      out_stream.close();
    }
    MPICHK(MPI_Barrier(MPI_COMM_WORLD));
    sycl_target->free();
    mesh->free();
  }
}

TEST(Benchmark, bandwidth_copy_REAL) {
  if (benchmark_enabled) {

    const int ndim = 2;
    std::vector<int> dims(ndim);
    dims[0] = 32;
    dims[1] = 32;

    auto mesh =
        std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, 1, 0);
    auto sycl_target =
        std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
    auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
    auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<REAL>("A"), 1),
                               ParticleProp(Sym<REAL>("B"), 1)};
    auto A =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
    const int cell_count = mesh->get_cell_count();

    const int comm_size = sycl_target->comm_pair.size_parent;
    const int comm_rank = sycl_target->comm_pair.rank_parent;
    const bool root = sycl_target->comm_pair.rank_parent == 0;
    const int Ntest = 400;

    std::ofstream out_stream;
    if (root) {
      out_stream = std::ofstream("benchmark_bandwidth_copy_REAL.csv");
    }

    if (root) {
      auto lambda_print = [&](auto &os) {
        os << "Size (Bytes), Bandwidth (GB/s), Time Taken (s)" << std::endl;
      };
      lambda_print(std::cout);
      lambda_print(out_stream);
    }
    for (int px = 0; px < 16; px++) {
      const std::size_t N =
          std::pow(static_cast<std::size_t>(2), static_cast<std::size_t>(px)) *
          dims[0] * dims[1];

      std::size_t start = 0;
      std::size_t end = 0;
      get_decomp_1d(comm_size, N, comm_rank, &start, &end);
      std::size_t Nlocal = end - start;

      A->clear();
      ParticleSet initial_distribution(Nlocal, particle_spec);
      for (std::size_t px = 0; px < Nlocal; px++) {
        initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
      }

      A->add_particles_local(initial_distribution);

      auto loop = particle_loop(
          A, [=](auto A, auto B) { A.at(0) = B.at(0); },
          Access::write(Sym<REAL>("A")), Access::read(Sym<REAL>("B")));

      MPICHK(MPI_Barrier(MPI_COMM_WORLD));
      auto t0 = profile_timestamp();

      for (int testx = 0; testx < Ntest; testx++) {
        loop->execute();
      }

      MPICHK(MPI_Barrier(MPI_COMM_WORLD));
      auto t1 = profile_timestamp();
      auto time_taken = profile_elapsed(t0, t1);

      const std::size_t num_bytes = N * 2 * sizeof(REAL);
      auto bw = Ntest * num_bytes / time_taken;
      bw /= 1e9;

      if (root) {
        auto lambda_print = [&](auto &os) {
          os << std::setfill(' ') << std::setw(12) << num_bytes << ","
             << std::setw(16) << std::scientific << bw << "," << std::setw(16)
             << std::scientific << time_taken << std::endl;
        };
        lambda_print(std::cout);
        lambda_print(out_stream);
      }
    }

    std::cout << std::flush;
    if (root) {
      out_stream << std::flush;
      out_stream.close();
    }
    MPICHK(MPI_Barrier(MPI_COMM_WORLD));
    sycl_target->free();
    mesh->free();
  }
}
