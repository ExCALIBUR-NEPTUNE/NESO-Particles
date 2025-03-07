#ifndef _NESO_PARTICLES_SUB_GROUP_SELECTOR_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SELECTOR_HPP_

#include "../containers/local_array.hpp"
#include "../containers/sym_vector.hpp"
#include "../loop/cell_info_npart.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../loop/particle_loop_index.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

/**
 * Class to consume the lambda which selects which particles are to be in a
 * ParticleSubGroup and provide to the ParticleSubGroup a list of cells and
 * layers.
 */
class SubGroupSelector : public SubGroupSelectorBase {
  friend class NESO::Particles::ParticleSubGroup;

protected:
  ParticleLoopSharedPtr loop_0;
  ParticleLoopSharedPtr loop_1;

  template <template <typename> typename T, typename U>
  inline void check_sym_type(T<U> arg) {
    static_assert(std::is_same<T<U>, Sym<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Sym type check failed.");

    // add this sym to the version checker signature
    this->particle_dat_versions[arg] = 0;
  }

  template <typename T> inline void check_sym_type(SymVectorSharedPtr<T> sv) {
    auto dats = sv->get_particle_dats();
    for (auto &dx : dats) {
      this->check_sym_type(dx->sym);
    }
  }

  inline void check_sym_type([[maybe_unused]] ParticleLoopIndex &) {}
  inline void check_sym_type([[maybe_unused]] CellInfoNPart &) {}

  template <template <typename> typename T, typename U>
  inline void check_read_access(T<U> arg) {
    static_assert(std::is_same<T<U>, Access::Read<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Read access check failed.");
    check_sym_type(arg.obj);
  }

  SubGroupSelector() = default;

  template <typename PARENT>
  inline void internal_setup(std::shared_ptr<PARENT> parent) {
    this->internal_setup_base();
    this->add_parent_dependencies(parent);
    this->particle_group_version = 0;

    this->loop_1 = particle_loop(
        "sub_group_selector_1", parent,
        [=](auto loop_index, auto k_map_cell_to_particles, auto k_map_ptrs) {
          const INT particle_linear_index = loop_index.get_local_linear_index();
          const int layer = k_map_ptrs.at(0)[particle_linear_index];
          const bool required = layer > -1;
          if (required) {
            k_map_cell_to_particles.at(layer, 0) = loop_index.layer;
          }
        },
        Access::read(ParticleLoopIndex{}),
        Access::write(this->map_cell_to_particles),
        Access::read(this->map_ptrs));
  }

  inline auto get_particle_group_sub_group_layers() {
    return particle_group->d_sub_group_layers;
  }

public:
  /**
   * Create a selector based on a kernel and arguments. The selector kernel
   * must be a lambda which returns true for particles which are in the sub
   * group and false for particles which are not in the sub group. The
   * arguments for the selector kernel must be read access Syms, i.e.
   * Access::read(Sym<T>("name")), ParticleLoopIndex or CellInfoNPart.
   *
   * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
   * ParticleSubGroup.
   * @param kernel Lambda function (like a ParticleLoop kernel) that returns
   * true for the particles which should be in the ParticleSubGroup.
   * @param args Arguments in the form of access descriptors wrapping objects
   * to pass to the kernel.
   */
  template <typename PARENT, typename KERNEL, typename... ARGS>
  SubGroupSelector(std::shared_ptr<PARENT> parent, KERNEL kernel, ARGS... args)
      : SubGroupSelectorBase(parent) {

    (check_read_access(args), ...);
    this->internal_setup(parent);

    this->loop_0 = particle_loop(
        "sub_group_selector_0", parent,
        [=](auto loop_index, auto k_map_ptrs, auto... user_args) {
          const bool required = kernel(user_args...);
          const INT particle_linear_index = loop_index.get_local_linear_index();
          if (required) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                element_atomic(k_map_ptrs.at(1)[loop_index.cell]);
            const int layer = element_atomic.fetch_add(1);
            k_map_ptrs.at(0)[particle_linear_index] = layer;
          } else {
            k_map_ptrs.at(0)[particle_linear_index] = -1;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(this->map_ptrs),
        args...);
  }

  virtual ~SubGroupSelector() = default;
  /**
   * Get two BufferDeviceHost objects that hold the cells and layers of the
   * particles which currently are selected by the selector kernel.
   *
   * @returns List of cells and layers of particles in the sub group.
   */
  virtual inline Selection get() override {
    const int cell_count = this->particle_group->domain->mesh->get_cell_count();
    auto sycl_target = this->particle_group->sycl_target;
    auto pg_map_layers = particle_group->d_sub_group_layers;
    int *d_npart_cell_ptr = this->dh_npart_cell->d_buffer.ptr;
    auto e0 = sycl_target->queue.fill<int>(d_npart_cell_ptr, 0, cell_count);
    const auto npart_local = this->particle_group->get_npart_local();
    pg_map_layers->realloc_no_copy(npart_local);

    std::vector<int *> tmp = {pg_map_layers->ptr, d_npart_cell_ptr};
    this->map_ptrs->set(tmp);
    e0.wait_and_throw();

    this->loop_0->execute();

    this->dh_npart_cell->device_to_host();
    int *h_npart_cell_ptr = this->dh_npart_cell->h_buffer.ptr;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const INT nrow_required = h_npart_cell_ptr[cellx];
      if (this->map_cell_to_particles->nrow.at(cellx) < nrow_required) {
        this->map_cell_to_particles->set_nrow(cellx, nrow_required);
      }
    }
    this->map_cell_to_particles->wait_set_nrow();

    this->loop_1->submit();

    std::vector<INT> h_npart_cell_es(cell_count);
    INT total = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      h_npart_cell_es[cellx] = total;
      total += h_npart_cell_ptr[cellx];
    }
    INT *d_npart_cell_es_ptr = this->d_npart_cell_es->ptr;
    sycl_target->queue
        .memcpy(d_npart_cell_es_ptr, h_npart_cell_es.data(),
                cell_count * sizeof(INT))
        .wait();

    this->loop_1->wait();

    Selection s;
    s.npart_local = total;
    s.ncell = cell_count;
    s.h_npart_cell = h_npart_cell_ptr;
    s.d_npart_cell = d_npart_cell_ptr;
    s.d_npart_cell_es = d_npart_cell_es_ptr;
    s.d_map_cells_to_particles = {this->map_cell_to_particles->device_ptr()};
    return s;
  }
};

typedef std::shared_ptr<SubGroupSelector> SubGroupSelectorSharedPtr;

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
