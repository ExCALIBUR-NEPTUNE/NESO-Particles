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
  SYCLTarget(const int gpu_device, MPI_Comm comm) : comm_pair(comm) {
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

template <typename T> class BufferDevice {
private:
public:
  SYCLTarget &sycl_target;
  T *ptr;
  size_t size;
  BufferDevice(SYCLTarget &sycl_target, size_t size)
      : sycl_target(sycl_target) {
    this->size = size;
    this->ptr = (T *)sycl::malloc_device(size * sizeof(T), sycl_target.queue);
  }
  inline int realloc_no_copy(const size_t size) {
    if (size > this->size) {
      sycl::free(this->ptr, this->sycl_target.queue);
      this->ptr = (T *)sycl::malloc_device(size * sizeof(T), sycl_target.queue);
      this->size = size;
    }
    return this->size;
  }
  ~BufferDevice() {
    if (this->ptr != NULL) {
      sycl::free(this->ptr, sycl_target.queue);
    }
  }
};

template <typename T> class BufferShared {
private:
public:
  SYCLTarget &sycl_target;
  T *ptr;
  size_t size;
  BufferShared(SYCLTarget &sycl_target, size_t size)
      : sycl_target(sycl_target) {
    this->size = size;
    this->ptr = (T *)sycl::malloc_shared(size * sizeof(T), sycl_target.queue);
  }
  inline int realloc_no_copy(const size_t size) {
    if (size > this->size) {
      sycl::free(this->ptr, this->sycl_target.queue);
      this->ptr = (T *)sycl::malloc_shared(size * sizeof(T), sycl_target.queue);
      this->size = size;
    }
    return this->size;
  }
  ~BufferShared() {
    if (this->ptr != NULL) {
      sycl::free(this->ptr, sycl_target.queue);
    }
  }
  // inline T& operator[](const int index) {
  //   return this->ptr[index];
  // };
};

template <typename T> class BufferHost {
private:
public:
  SYCLTarget &sycl_target;
  T *ptr;
  size_t size;
  BufferHost(SYCLTarget &sycl_target, size_t size) : sycl_target(sycl_target) {
    this->size = size;
    this->ptr = (T *)sycl::malloc_host(size * sizeof(T), sycl_target.queue);
  }
  inline int realloc_no_copy(const size_t size) {
    if (size > this->size) {
      sycl::free(this->ptr, this->sycl_target.queue);
      this->ptr = (T *)sycl::malloc_host(size * sizeof(T), sycl_target.queue);
      this->size = size;
    }
    return this->size;
  }
  ~BufferHost() {
    if (this->ptr != NULL) {
      sycl::free(this->ptr, sycl_target.queue);
    }
  }
};

} // namespace NESO::Particles

#endif
