#ifndef _NESO_PARTICLES_COMMUNICATION_GLOBAL_KEY_VALUE_MAP_NODE_WISE_HPP_
#define _NESO_PARTICLES_COMMUNICATION_GLOBAL_KEY_VALUE_MAP_NODE_WISE_HPP_

#include "../compute_target.hpp"
#include "communication_typedefs.hpp"
#include <limits>
#include <map>
#include <numeric>

namespace NESO::Particles {

/**
 * This is a key value map for keys in a given range. Values are stored
 * node-wise where nodes are determined by MPI shared memory split. The map is
 * consistent globally on the entire MPI communicator.
 */
template <typename KEY_TYPE, typename VALUE_TYPE>
class GlobalKeyValueLUTNodewise {
protected:
  bool is_allocated{false};

  MPI_Win map_offsets_win;
  std::size_t *map_offsets;
  std::size_t *map_offsets_base;

  MPI_Win map_data_win;
  std::byte *map_data;
  std::byte *map_data_base;

public:
  ~GlobalKeyValueLUTNodewise() {
    if (this->is_allocated) {
      nprint("Warning: GlobalKeyValueLUTNodewise::free not called.");
    }
  }

  // The SYCLTarget providing the MPI communicators.
  SYCLTargetSharedPtr sycl_target;
  // The start of the range of keys [key_start, key_end).
  KEY_TYPE key_start;
  // The end of the range of keys [key_start, key_end).
  KEY_TYPE key_end;

  /**
   * Construct a look-up-table (LUT) from passed values. Must be called
   * collectively on the communicator.
   *
   * Values of key_value_data where the key is present on multiple ranks must
   * provide the exact same value for the key.
   *
   * @param sycl_target Compute device providing the MPI communicator.
   * @param key_value_data Map from keys to values to place in the LUT.
   */
  GlobalKeyValueLUTNodewise(SYCLTargetSharedPtr sycl_target,
                            std::map<KEY_TYPE, VALUE_TYPE> &key_value_data)
      : is_allocated(true), sycl_target(sycl_target) {

    KEY_TYPE key_start_local = std::numeric_limits<KEY_TYPE>::max();
    KEY_TYPE key_end_local = std::numeric_limits<KEY_TYPE>::lowest();

    for (auto &m : key_value_data) {
      const KEY_TYPE key = m.first;
      key_start_local = std::min(key_start_local, key);
      key_end_local = std::max(key_start_local, key + 1);
    }

    const auto mpi_key_type = map_ctype_mpi_type<KEY_TYPE>();

    const KEY_TYPE num_elements_local = key_value_data.size();
    KEY_TYPE num_elements_global = 0;

    MPICHK(MPI_Allreduce(&num_elements_local, &num_elements_global, 1,
                         mpi_key_type, MPI_SUM,
                         this->sycl_target->comm_pair.comm_parent));

    if (num_elements_global) {

      MPICHK(MPI_Allreduce(&key_start_local, &this->key_start, 1, mpi_key_type,
                           MPI_MIN, this->sycl_target->comm_pair.comm_parent));

      MPICHK(MPI_Allreduce(&key_end_local, &this->key_end, 1, mpi_key_type,
                           MPI_MAX, this->sycl_target->comm_pair.comm_parent));

    } else {
      this->key_start = 0;
      this->key_end = 0;
    }
    auto &comm_pair = sycl_target->comm_pair;

    const std::size_t offsets_len = (this->key_end - this->key_start) + 1;
    const MPI_Aint num_alloc_offsets =
        (comm_pair.rank_intra == 0) ? offsets_len * sizeof(std::size_t) : 0;
    MPICHK(MPI_Win_allocate_shared(num_alloc_offsets, sizeof(std::size_t),
                                   MPI_INFO_NULL, comm_pair.comm_intra,
                                   (void *)&this->map_offsets,
                                   &this->map_offsets_win));
    MPI_Aint tmp_size = 0;
    int tmp_disp_unit = 0;
    MPICHK(MPI_Win_shared_query(this->map_offsets_win, 0, &tmp_size,
                                &tmp_disp_unit, &this->map_offsets_base));

    std::size_t win_start = 0;
    std::size_t win_end = 0;
    get_decomp_1d(static_cast<std::size_t>(comm_pair.size_intra), offsets_len,
                  static_cast<std::size_t>(comm_pair.rank_intra), &win_start,
                  &win_end);

    MPICHK(MPI_Win_fence(0, this->map_offsets_win));
    std::fill(this->map_offsets_base + win_start,
              this->map_offsets_base + win_end, 0);
    MPICHK(MPI_Win_fence(0, this->map_offsets_win));

    MPICHK(MPI_Win_fence(0, this->map_offsets_win));
    std::size_t max_local_size = 0;
    for (auto &m : key_value_data) {
      const KEY_TYPE key = m.first - this->key_start;
      const std::size_t size = m.second.get_num_bytes();
      this->map_offsets_base[key] = size;
      max_local_size = std::max(max_local_size, size);
    }
    MPICHK(MPI_Win_fence(0, this->map_offsets_win));

    MPICHK(MPI_Win_fence(0, this->map_offsets_win));
    if (comm_pair.rank_intra == 0) {

      std::vector<std::size_t> tmp_map_offsets(offsets_len);
      std::fill(tmp_map_offsets.begin(), tmp_map_offsets.end(), 0);

      MPICHK(MPI_Allreduce(this->map_offsets_base, tmp_map_offsets.data(),
                           offsets_len, get_size_t_mpi_type(), MPI_MAX,
                           comm_pair.comm_inter));

      std::exclusive_scan(tmp_map_offsets.begin(), tmp_map_offsets.end(),
                          this->map_offsets_base, 0);
    }
    MPICHK(MPI_Win_fence(0, this->map_offsets_win));

    const std::size_t total_num_bytes = this->map_offsets_base[offsets_len - 1];

    const MPI_Aint num_alloc_data =
        (comm_pair.rank_intra == 0) ? total_num_bytes : 0;
    MPICHK(MPI_Win_allocate_shared(
        num_alloc_data, sizeof(std::byte), MPI_INFO_NULL, comm_pair.comm_intra,
        (void *)&this->map_data, &this->map_data_win));
    MPICHK(MPI_Win_shared_query(this->map_data_win, 0, &tmp_size,
                                &tmp_disp_unit, &this->map_data_base));

    std::size_t byte_start = 0;
    std::size_t byte_end = 0;
    get_decomp_1d(
        static_cast<std::size_t>(comm_pair.size_intra), total_num_bytes,
        static_cast<std::size_t>(comm_pair.rank_intra), &byte_start, &byte_end);

    MPICHK(MPI_Win_fence(0, this->map_data_win));
    {
      std::byte z{0};
      z = z ^ z;
      std::fill(this->map_data_base + byte_start,
                this->map_data_base + byte_end, z);
    }
    MPICHK(MPI_Win_fence(0, this->map_data_win));

    MPICHK(MPI_Win_fence(0, this->map_data_win));
    MPICHK(MPI_Win_fence(0, this->map_offsets_win));
    for (auto &m : key_value_data) {
      const KEY_TYPE key = m.first - this->key_start;
      const std::size_t offset = this->map_offsets_base[key];
      const std::size_t offset_next = this->map_offsets_base[key + 1];
      const std::size_t size = offset_next - offset;
      m.second.serialise(this->map_data_base + offset, size);
    }
    MPICHK(MPI_Win_fence(0, this->map_offsets_win));
    MPICHK(MPI_Win_fence(0, this->map_data_win));

    MPICHK(MPI_Win_fence(0, this->map_data_win));
    if (comm_pair.rank_intra == 0) {
      std::vector<std::size_t> tmp_map_data(total_num_bytes);
      std::fill(tmp_map_data.begin(), tmp_map_data.end(), 0);
      MPICHK(MPI_Allreduce(this->map_data_base, tmp_map_data.data(),
                           total_num_bytes, MPI_BYTE, MPI_BOR,
                           comm_pair.comm_inter));
      std::memcpy(this->map_data_base, tmp_map_data.data(), total_num_bytes);
    }
    MPICHK(MPI_Win_fence(0, this->map_data_win));
    MPICHK(MPI_Barrier(comm_pair.comm_parent));
  }

  /**
   * Retrieve values. Must be called collectively on the communicator.
   *
   * @param[in] keys Set of keys to retrieve value for.
   * @returns Deserialised values.
   */
  inline std::map<KEY_TYPE, VALUE_TYPE> get(const std::set<KEY_TYPE> keys) {

    for (auto &key : keys) {
      NESOASSERT((this->key_start <= key) && (key < this->key_end),
                 "Invalid key passed.");
    }

    std::map<KEY_TYPE, VALUE_TYPE> values;

    MPICHK(MPI_Win_fence(0, this->map_data_win));
    MPICHK(MPI_Win_fence(0, this->map_offsets_win));
    for (auto &key : keys) {
      const KEY_TYPE key_origin = key - this->key_start;
      const std::size_t offset = this->map_offsets_base[key_origin];
      const std::size_t offset_next = this->map_offsets_base[key_origin + 1];
      const std::size_t size = offset_next - offset;
      values[key].deserialise(this->map_data_base + offset, size);
    }

    MPICHK(MPI_Win_fence(0, this->map_offsets_win));
    MPICHK(MPI_Win_fence(0, this->map_data_win));
    return values;
  }

  /**
   * Free the key-value map. Must be called. Must be called collectively.
   */
  inline void free() {
    if (this->is_allocated) {
      MPICHK(MPI_Win_free(&this->map_offsets_win))
      this->map_offsets = nullptr;
      this->map_offsets_base = nullptr;
      MPICHK(MPI_Win_free(&this->map_data_win))
      this->map_data = nullptr;
      this->map_data_base = nullptr;
    }
    this->is_allocated = false;
  }
};

} // namespace NESO::Particles

#endif
