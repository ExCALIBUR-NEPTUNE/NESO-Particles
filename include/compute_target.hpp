#ifndef _NESO_PARTICLES_COMPUTE_TARGET
#define _NESO_PARTICLES_COMPUTE_TARGET

#include <CL/sycl.hpp>
#include <mpi.h>

#include "communication.hpp"
#include "typedefs.hpp"

using namespace cl;

namespace NESO::Particles {

class SYCLTarget {
private:
public:
  sycl::device device;
  sycl::queue queue;
  MPI_Comm comm;
  CommPair comm_pair;

  SYCLTarget(){};
  SYCLTarget(const int gpu_device, MPI_Comm comm) : comm_pair(comm_pair) {
    if (gpu_device > 0) {
      try {
        this->device = sycl::device(sycl::gpu_selector());
      } catch (sycl::exception const &e) {
        std::cout << "Cannot select a GPU\n" << e.what() << "\n";
        std::cout << "Using a CPU device\n";
        this->device = sycl::device(sycl::cpu_selector());
      }
    } else if (gpu_device < 0) {
      this->device = sycl::device(sycl::cpu_selector());
    } else {
      this->device = sycl::device(sycl::default_selector());
    }

    std::cout << "Using " << this->device.get_info<sycl::info::device::name>()
              << std::endl;

    this->queue = sycl::queue(this->device);
    this->comm = comm;
  }
  ~SYCLTarget() {}

  void free() { comm_pair.free(); }
};

} // namespace NESO::Particles

#endif
