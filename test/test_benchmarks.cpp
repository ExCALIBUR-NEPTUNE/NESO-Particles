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

static const std::size_t benchmark_size =
    get_env_size_t("NESO_PARTICLES_BENCHMARK_SIZE", 0);

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

    const int s = benchmark_size ? static_cast<int>(benchmark_size) : 31;
    for (int px = 0; px < s; px++) {
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
    const std::size_t cell_count_global = dims[0] * dims[1];

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

    const bool root = sycl_target->comm_pair.rank_parent == 0;
    const int Ntest = 800;

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
    const int s = benchmark_size ? static_cast<int>(benchmark_size) : 16;
    for (int px = 0; px < s; px++) {
      const std::size_t N_per_cell =
          std::pow(static_cast<std::size_t>(2), static_cast<std::size_t>(px));
      const std::size_t N = N_per_cell * cell_count_global;
      std::size_t Nlocal = cell_count * N_per_cell;

      A->clear();
      ParticleSet initial_distribution(Nlocal, particle_spec);
      for (std::size_t px = 0; px < Nlocal; px++) {
        initial_distribution.at(Sym<INT>("CELL_ID"), px, 0) = px % cell_count;
      }

      A->add_particles_local(initial_distribution);

      auto loop = particle_loop(
          A, [=](auto A, auto B) { A.at(0) = B.at(0) + B.at(1) * B.at(2); },
          Access::write(Sym<REAL>("A")), Access::read(Sym<REAL>("B")));

      for (int testx = 0; testx < 10; testx++) {
        loop->execute();
      }

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

TEST(Benchmark, bandwidth_copy_REAL) {
  if (benchmark_enabled) {

    const int ndim = 2;
    std::vector<int> dims(ndim);
    dims[0] = 32;
    dims[1] = 32;
    const std::size_t cell_count_global = dims[0] * dims[1];

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
    const bool root = sycl_target->comm_pair.rank_parent == 0;
    const int Ntest = 800;

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

    const int s = benchmark_size ? static_cast<int>(benchmark_size) : 16;
    for (int px = 0; px < s; px++) {
      const std::size_t N_per_cell =
          std::pow(static_cast<std::size_t>(2), static_cast<std::size_t>(px));
      const std::size_t N = N_per_cell * cell_count_global;
      std::size_t Nlocal = cell_count * N_per_cell;

      A->clear();
      ParticleSet initial_distribution(Nlocal, particle_spec);
      for (std::size_t px = 0; px < Nlocal; px++) {
        initial_distribution.at(Sym<INT>("CELL_ID"), px, 0) = px % cell_count;
      }

      A->add_particles_local(initial_distribution);

      auto loop = particle_loop(
          A, [=](auto A, auto B) { A.at(0) = B.at(0); },
          Access::write(Sym<REAL>("A")), Access::read(Sym<REAL>("B")));

      for (int testx = 0; testx < 10; testx++) {
        loop->execute();
      }

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

TEST(Benchmark, flops_fma_REAL) {
  if (benchmark_enabled) {

    const int ndim = 2;
    constexpr int mdim = 4;
    std::vector<int> dims(ndim);
    dims[0] = 32;
    dims[1] = 32;
    const std::size_t cell_count_global = dims[0] * dims[1];

    auto mesh =
        std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims, 1, 0);
    auto sycl_target =
        std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
    auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
    auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<REAL>("X"), mdim),
                               ParticleProp(Sym<REAL>("FX"), mdim)};
    auto A =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
    const int cell_count = mesh->get_cell_count();

    const bool root = sycl_target->comm_pair.rank_parent == 0;
    const int Ntest = 400;

    const std::size_t N_per_cell = 2024;
    const std::size_t N = N_per_cell * cell_count_global;
    std::size_t Nlocal = cell_count * N_per_cell;

    ParticleSet initial_distribution(Nlocal, particle_spec);

    REAL x = 0.1;
    REAL xx = 7.123127;
    for (std::size_t px = 0; px < Nlocal; px++) {
      initial_distribution.at(Sym<INT>("CELL_ID"), px, 0) = px % cell_count;
      for (int dx = 0; dx < mdim; dx++) {
        initial_distribution.at(Sym<REAL>("X"), px, dx) = std::fmod(x, 1.0);
        x += xx;
      }
    }

    A->add_particles_local(initial_distribution);

    std::ofstream out_stream;
    if (root) {
      out_stream = std::ofstream("benchmark_flops_fma_REAL.csv");
    }

    if (root) {
      auto lambda_print = [&](auto &os) {
        os << "Order, Bandwidth (GB/s), FLOPs (GFLOP/s), Time Taken (s)"
           << std::endl;
      };
      lambda_print(std::cout);
      lambda_print(out_stream);
    }

    const int s = benchmark_size ? static_cast<int>(benchmark_size) : 11;
    for (int pxx = 0; pxx < s; pxx++) {
      const int px = std::pow(2, pxx);

      // auto cdc_coeffs = std::make_shared<CellDatConst<REAL>>(
      //     sycl_target, cell_count, px, mdim);

      auto cdc_coeffs = std::make_shared<CellDatConst<REAL>>(
          sycl_target, cell_count, mdim, px);

      auto coeffs = cdc_coeffs->get_all_cells();
      for (int cellx = 0; cellx < cell_count; cellx++) {
        for (int cx = 0; cx < mdim; cx++) {
          for (int rx = 0; rx < px; rx++) {
            // coeffs.at(cellx)->at(rx, cx) = std::fmod(x, 1.0);
            coeffs.at(cellx)->at(cx, rx) = std::fmod(x, 1.0);
            x += xx;
          }
        }
      }
      cdc_coeffs->set_all_cells(coeffs);

      auto loop = particle_loop(
          A,
          [=](auto X, auto FX, auto COEFFS) {
            // REAL fx = 0.0;
            // for(int dx=0 ; dx<mdim ; dx++){
            //   const REAL x = X.at(dx);
            //   REAL fx = 0.0;
            //   for(int ox=0 ; ox<px ; ox++){
            //      fx = Kernel::fma(x, fx, COEFFS.at(ox, dx));
            //     //fx = Kernel::fma(x, fx, COEFFS.at(dx, ox));
            //   }
            //   FX.at(dx) = fx;
            // }

            REAL fx[mdim];
            REAL x[mdim];

            for (int dx = 0; dx < mdim; dx++) {
              fx[dx] = 0.0;
              x[dx] = X.at(dx);
            }

            for (int ox = 0; ox < px; ox++) {
              for (int dx = 0; dx < mdim; dx++) {
                // fx[dx] = Kernel::fma(x[dx], fx[dx], COEFFS.at(ox, dx));
                fx[dx] = Kernel::fma(x[dx], fx[dx], COEFFS.at(dx, ox));
              }
            }

            for (int dx = 0; dx < mdim; dx++) {
              FX.at(dx) = fx[dx];
            }
          },
          Access::read(Sym<REAL>("X")), Access::write(Sym<REAL>("FX")),
          Access::read(cdc_coeffs));

      for (int testx = 0; testx < 10; testx++) {
        loop->execute();
      }

      MPICHK(MPI_Barrier(MPI_COMM_WORLD));
      auto t0 = profile_timestamp();

      for (int testx = 0; testx < Ntest; testx++) {
        loop->execute();
      }

      MPICHK(MPI_Barrier(MPI_COMM_WORLD));
      auto t1 = profile_timestamp();
      auto time_taken = profile_elapsed(t0, t1);

      const std::size_t num_bytes = N * mdim * 2 * sizeof(REAL);
      auto bw = Ntest * num_bytes / time_taken;
      bw /= 1e9;

      const std::size_t num_flops = N * mdim * px * 2;
      auto fr = Ntest * num_flops / time_taken;
      fr /= 1e9;

      if (root) {
        auto lambda_print = [&](auto &os) {
          os << std::setfill(' ') << std::setw(12) << px << "," << std::setw(16)
             << std::scientific << bw << "," << std::setw(16) << std::scientific
             << fr << "," << std::setw(16) << std::scientific << time_taken
             << std::endl;
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
