#ifndef __NESO_PARTICLES_EPHEMERAL_EPHEMERAL_DAT_HPP_
#define __NESO_PARTICLES_EPHEMERAL_EPHEMERAL_DAT_HPP_

#include "../../particle_dat.hpp"

namespace NESO::Particles {
class EphemeralDats;

template <typename T> class EphemeralDat {
  friend class EphemeralDats;

protected:
  std::shared_ptr<BufferDevice<T **>> d_cell_ptrs;
  std::shared_ptr<BufferDevice<T *>> d_col_ptrs;
  std::shared_ptr<BufferDevice<T>> d_data;
  T ***d_ptr;

  inline ParticleDatImplGetT<T> impl_get() { return this->d_ptr; }
  inline ParticleDatImplGetConstT<T> impl_get_const() { return this->d_ptr; }

  EphemeralDat(std::shared_ptr<BufferDevice<T **>> d_cell_ptrs,
               std::shared_ptr<BufferDevice<T *>> d_col_ptrs,
               std::shared_ptr<BufferDevice<T>> d_data)
      : d_cell_ptrs(d_cell_ptrs), d_col_ptrs(d_col_ptrs), d_data(d_data),
        d_ptr(d_cell_ptrs->ptr) {}

public:
  ~EphemeralDat() = default;
};

template <typename T>
using EphemeralDatSharedPtr = std::shared_ptr<EphemeralDat<T>>;

} // namespace NESO::Particles

#endif
