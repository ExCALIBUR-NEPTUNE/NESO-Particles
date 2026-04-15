#include <gtest/gtest.h>

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
        os << "Size (Bytes), Host To Device (GB/s), Device To Host (GB/s)"
           << std::endl;
      };
      lambda_print(std::cout);
      lambda_print(out_stream);
    }
    for (int px = 0; px < 31; px++) {
      const std::size_t N =
          std::pow(static_cast<std::size_t>(2), static_cast<std::size_t>(px));

      std::vector<unsigned char> h_buffer(N);
      std::fill(h_buffer.begin(), h_buffer.end(), 1);
      BufferDevice<unsigned char> d_buffer(sycl_target, h_buffer);

      auto lambda_do_run = [&](const bool to_device) -> REAL {
        MPICHK(MPI_Barrier(MPI_COMM_WORLD));
        auto t0 = profile_timestamp();
        for (int testx = 0; testx < Ntest; testx++) {
          if (to_device) {
            sycl_target->queue.memcpy(d_buffer.ptr, h_buffer.data(), N).wait();
          } else {
            sycl_target->queue.memcpy(h_buffer.data(), d_buffer.ptr, N).wait();
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

      const REAL host_to_device = lambda_do_run(true);
      const REAL device_to_host = lambda_do_run(false);

      if (root) {
        auto lambda_print = [&](auto &os) {
          os << std::setfill(' ') << std::setw(12) << N << "," << std::setw(16)
             << std::scientific << host_to_device << "," << std::setw(16)
             << std::scientific << device_to_host << "" << std::endl;
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
