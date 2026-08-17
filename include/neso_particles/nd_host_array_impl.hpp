#ifndef _NESO_PARTICLES_ND_HOST_ARRAY_IMPL_H_
#define _NESO_PARTICLES_ND_HOST_ARRAY_IMPL_H_

#include "containers/nd_local_array.hpp"
#include "nd_host_array.hpp"

namespace NESO::Particles {

template <typename T, std::size_t N>
NDHostArray<T, N>::NDHostArray(
    SYCLTargetSharedPtr sycl_target,
    std::shared_ptr<NDLocalArray<T, N>> &nd_local_array)
    : NDHostArray(sycl_target, nd_local_array->index) {

  this->sycl_target->queue
      .memcpy(this->ptr(), nd_local_array->ptr(),
              this->index.size() * sizeof(T))
      .wait_and_throw();
}

} // namespace NESO::Particles

#endif
