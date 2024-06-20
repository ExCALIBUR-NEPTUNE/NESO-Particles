#ifndef _NESO_PARTICLES_MESH_HIERARCHY_CONTAINER_HPP_
#define _NESO_PARTICLES_MESH_HIERARCHY_CONTAINER_HPP_

#include "../communication.hpp"
#include "../mesh_hierarchy.hpp"
#include "../typedefs.hpp"
#include "serial_container.hpp"
#include "serial_interface.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>

namespace NESO::Particles::MeshHierarchyData {

/**
 * A repeated pattern in this library is to collect all objects of a certain
 * type that touch a particular @ref MeshHierarchy cell on the MPI rank that
 * owns the cell. Later other MPI ranks request all of these objects for a
 * particular cell.
 *
 * For example to construct the halo regions we collect all mesh cells that
 * intersect a MH cell on the rank that owns the cell. MPI ranks then request
 * these cells for all the MH cells that they need to construct the halos.
 *
 * A second example would be computing the intersection of particle
 * trajectories and the mesh boundary. We collect the boundary elements on the
 * MH cell that intersects the boundary. MPI ranks then pull from the rank that
 * owns the MH cell all boundary elements that touch the MH cell.
 *
 * For a type to be compatible with this class an interface must be defined by
 * creating a class which inherits from @ref
 * MeshHierarchyData::SerialInterface.
 *
 * The process to use this class is as follows:
 * 1) On construction pass a map from MH cells to a std::vector of serialisable
 * objects. 2) Call gather collectively with a vector of cells the MPI rank
 * requires. 3) Call get to obtain a std::vector of the deserialsed objects for
 * a cell.
 */
template <typename T> class MeshHierarchyContainer {
protected:
  /**
   * This class will collect SerialInterface derived types onto an MPI rank.
   * These collections are need to be serialised then sent to the owning rank.
   * On the owning rank these buffers are concatenated into a single contiguous
   * buffer that holds all the information for each cell.
   *
   * When a remote rank requests the cell it is served the serialised buffer
   * that contains all the objects. The remote rank then deserialises all the
   * objects to return them to the user.
   *
   * The serialisation/deserialisation/concatenation of these intermediate
   * buffers is also described by the SerialInterface and that is what this
   * helper type implements.
   */
  struct MHCellBuffer : public SerialInterface {
    INT cell;
    std::vector<std::byte> buffer;

    MHCellBuffer() = default;
    MHCellBuffer(INT cell, std::vector<std::byte> &buffer)
        : cell(cell), buffer(buffer){};

    virtual inline std::size_t get_num_bytes() const override {
      return sizeof(INT) + sizeof(std::size_t) + this->buffer.size();
    }
    virtual inline void serialise(std::byte *buffer,
                                  const std::size_t num_bytes) const override {
      const std::byte *check_end = buffer + num_bytes;
      std::memcpy(buffer, &this->cell, sizeof(INT));
      buffer += sizeof(INT);
      const std::size_t buffer_size = this->buffer.size();
      std::memcpy(buffer, &buffer_size, sizeof(std::size_t));
      buffer += sizeof(size_t);
      std::memcpy(buffer, this->buffer.data(), buffer_size);
      NESOASSERT(buffer + buffer_size == check_end,
                 "Packing under/overflow occured.");
    }
    virtual inline void deserialise(const std::byte *buffer,
                                    const std::size_t num_bytes) override {
      const std::byte *check_end = buffer + num_bytes;
      std::memcpy(&this->cell, buffer, sizeof(INT));
      buffer += sizeof(INT);
      std::size_t buffer_size = std::numeric_limits<std::size_t>::max();
      std::memcpy(&buffer_size, buffer, sizeof(std::size_t));
      buffer += sizeof(std::size_t);
      NESOASSERT(buffer_size < std::numeric_limits<std::size_t>::max(),
                 "Unexpected buffer size.");
      this->buffer.resize(buffer_size);
      std::memcpy(this->buffer.data(), buffer, buffer_size);
      NESOASSERT(buffer + buffer_size == check_end,
                 "Packing under/overflow occured.");
    }
  };

  MPI_Comm comm;
  CommunicationEdgesCounter ece;
  std::map<int, SerialContainer<T>> map_cell_buffers;
  std::set<INT> cells_gathered;

  inline void generic_unpack(
      std::map<int, SerialContainer<MHCellBuffer>> &map_incoming_buffers) {
    std::vector<MHCellBuffer> tmp;
    // unpack the rank wise data into cell wise data
    for (auto &item : map_incoming_buffers) {
      auto &buffer = item.second;
      buffer.get(tmp);
      for (auto &tx : tmp) {
        const INT cell = tx.cell;
        auto &inner_buffer = tx.buffer;
        SerialContainer<T> tmp_serial(inner_buffer.size());
        std::memcpy(tmp_serial.buffer.data(), inner_buffer.data(),
                    inner_buffer.size());
        this->map_cell_buffers[cell].append(tmp_serial);
        this->cells_gathered.insert(cell);
      }
    }
  }

public:
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;

  /**
   * Get the deserialised objects for all objects related to a particular cell.
   * The gather method must have been called with the requested cell prior to
   * calling get. This method is NOT collective on the communicator.
   *
   * @param[in] cell MeshHierarchy cell to collect objects for.
   * @param[in, out] output Container to place objects in.
   */
  inline void get(const INT cell, std::vector<T> &output) {
    NESOASSERT(this->map_cell_buffers.count(cell),
               "Cannot get cell with gather has not been called for.");
    this->map_cell_buffers.at(cell).get(output);
  }

  /**
   * Locally gather on this MPI rank all the serialised objects which are
   * stored on the passed MeshHierarchy cells. This method is collective on the
   * communicator.
   *
   * @param cells MeshHierarchy cells for which to gather objects.
   */
  inline void gather(const std::vector<INT> &cells) {
    int mpi_size;
    MPICHK(MPI_Comm_size(this->comm, &mpi_size));

    std::set<INT> cells_set;
    for (const INT &cell : cells) {
      if (!this->cells_gathered.count(cell)) {
        cells_set.insert(cell);
      }
    }

    // If no ranks actually require data return;
    const int num_cells = cells_set.size();
    int num_max_cells = -1;
    MPICHK(MPI_Allreduce(&num_cells, &num_max_cells, 1, MPI_INT, MPI_MAX,
                         this->comm));
    if (num_max_cells == 0) {
      return;
    }

    // collect the requested cells rank-wise
    std::map<int, std::vector<INT>> map_send_ranks_cells;
    std::vector<int> send_ranks;
    for (const INT &cell : cells_set) {
      const int rank = mesh_hierarchy->get_owner(cell);
      if ((-1 < rank) && (rank < mpi_size)) {
        send_ranks.push_back(rank);
        map_send_ranks_cells[rank].push_back(cell);
      }
    }

    // allocate space to send the requested cells to the ranks that own the
    // requested cells.
    const int num_send_ranks = send_ranks.size();
    std::vector<std::int64_t> send_num_cells_data;
    send_num_cells_data.reserve(num_send_ranks);
    std::vector<void *> send_cells_data;
    send_cells_data.reserve(num_send_ranks);
    for (const int &rank : send_ranks) {
      send_num_cells_data.push_back(map_send_ranks_cells.at(rank).size() *
                                    sizeof(INT));
      send_cells_data.push_back(map_send_ranks_cells.at(rank).data());
    }

    // tell the remote ranks this rank will request cells and send how many
    // cells will be requested
    std::vector<int> recv_ranks;
    std::vector<std::int64_t> recv_num_cells_data;
    this->ece.get_remote_ranks(send_ranks, send_num_cells_data, recv_ranks,
                               recv_num_cells_data);

    // allocate space for the incoming cell indices that are requested
    const int num_recv_ranks = recv_ranks.size();
    std::map<int, std::vector<INT>> map_recv_ranks_cells;
    std::vector<void *> recv_cells_data;
    recv_cells_data.reserve(num_recv_ranks);
    for (int rx = 0; rx < num_recv_ranks; rx++) {
      const int rank = recv_ranks.at(rx);
      const size_t num_recv_cells = recv_num_cells_data.at(rx) / sizeof(INT);
      map_recv_ranks_cells[rank] = std::vector<INT>(num_recv_cells);
      recv_cells_data.push_back(map_recv_ranks_cells.at(rank).data());
    }

    // exchange the actual cell indices which are requested
    this->ece.exchange_send_recv_data(send_ranks, send_num_cells_data,
                                      send_cells_data, recv_ranks,
                                      recv_num_cells_data, recv_cells_data);

    // pack the locally owned data that has been requested by a remote rank
    std::map<int, SerialContainer<MHCellBuffer>> map_rank_buffers;
    for (int rx = 0; rx < num_recv_ranks; rx++) {
      const int rank = recv_ranks.at(rx);
      auto &recv_cells = map_recv_ranks_cells.at(rank);
      for (INT &cell : recv_cells) {
        std::vector<MHCellBuffer> tmp_cell;
        tmp_cell.emplace_back(cell, this->map_cell_buffers[cell].buffer);
        SerialContainer<MHCellBuffer> tmp_packed_cell(tmp_cell);
        // append the data on the store for the owning rank
        map_rank_buffers[rank].append(tmp_packed_cell);
      }
      recv_num_cells_data.at(rx) =
          static_cast<std::int64_t>(map_rank_buffers[rank].buffer.size());
      recv_cells_data.at(rx) =
          static_cast<void *>(map_rank_buffers[rank].buffer.data());
    }

    // exchange the size of the requested data
    this->ece.exchange_send_recv_counts(recv_ranks, recv_num_cells_data,
                                        send_ranks, send_num_cells_data, true);

    // Allocate space for the incoming serialised data per rank
    std::map<int, SerialContainer<MHCellBuffer>> map_incoming_buffers;
    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int rank = send_ranks.at(rankx);
      const std::size_t num_bytes =
          static_cast<std::size_t>(send_num_cells_data.at(rankx));
      map_incoming_buffers[rank] = SerialContainer<MHCellBuffer>(num_bytes);
      send_cells_data.at(rankx) = map_incoming_buffers.at(rank).buffer.data();
    }

    // exchange serialised data
    this->ece.exchange_send_recv_data(recv_ranks, recv_num_cells_data,
                                      recv_cells_data, send_ranks,
                                      send_num_cells_data, send_cells_data);

    generic_unpack(map_incoming_buffers);
  }

  /**
   * Initialise the container by passing a map from MeshHierarchy cell indices
   * to a vector of serialisable objects that this rank holds. This
   * initialisation is collective on the communicator. The serialisable type
   * should inherit from SerialInterface.
   *
   * @param mesh_hierarchy MeshHierarchy instance used for this data structure.
   * @param data Map from MH cells to serialisable instances.
   */
  MeshHierarchyContainer(std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                         std::map<INT, std::vector<T>> &data)
      : mesh_hierarchy(mesh_hierarchy), comm(mesh_hierarchy->comm),
        ece(mesh_hierarchy->comm) {

    // map from rank to packed MHCellBuffers
    std::map<int, SerialContainer<MHCellBuffer>> map_rank_buffers;
    std::set<int> ranks_set;

    int mpi_size;
    MPICHK(MPI_Comm_size(this->comm, &mpi_size));

    for (auto &item : data) {
      const INT cell = item.first;
      // get the rank which owns the cell
      const int rank = mesh_hierarchy->get_owner(cell);
      NESOASSERT((-1 < rank) && (rank < mpi_size),
                 "No owner found for MeshHierarchy cell.");
      // pack the original vector
      SerialContainer<T> cell_contents(item.second);
      std::vector<MHCellBuffer> tmp_cell;
      tmp_cell.emplace_back(cell, cell_contents.buffer);
      SerialContainer<MHCellBuffer> tmp_packed_cell(tmp_cell);
      // append the data on the store for the owning rank
      map_rank_buffers[rank].append(tmp_packed_cell);
      // record this rank as one of interest
      ranks_set.insert(rank);
    }

    // get ranks we will send to as a vector
    std::vector<int> recv_ranks;
    std::vector<std::int64_t> recv_counts;
    std::vector<void *> recv_ptrs;

    const int num_recv_ranks = ranks_set.size();
    recv_ranks.reserve(num_recv_ranks);
    recv_counts.reserve(num_recv_ranks);
    recv_ptrs.reserve(num_recv_ranks);
    for (auto rx : ranks_set) {
      recv_ranks.push_back(rx);
      recv_counts.push_back(map_rank_buffers.at(rx).buffer.size());
      recv_ptrs.push_back(map_rank_buffers.at(rx).buffer.data());
    }

    // communicate to remote ranks that this rank will send data and find ranks
    // that will send data to this rank.
    std::vector<int> send_ranks;
    std::vector<std::int64_t> send_counts;
    std::vector<void *> send_ptrs;
    this->ece.get_remote_ranks(recv_ranks, recv_counts, send_ranks,
                               send_counts);
    const int num_send_ranks = send_ranks.size();
    send_ptrs.reserve(num_send_ranks);

    // Allocate space for the incoming serialised data per rank
    std::map<int, SerialContainer<MHCellBuffer>> map_incoming_buffers;
    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int rank = send_ranks.at(rankx);
      const std::size_t num_bytes =
          static_cast<std::size_t>(send_counts.at(rankx));
      map_incoming_buffers[rank] = SerialContainer<MHCellBuffer>(num_bytes);
      send_ptrs.push_back(map_incoming_buffers.at(rank).buffer.data());
    }

    // exchange serialised data
    this->ece.exchange_send_recv_data(recv_ranks, recv_counts, recv_ptrs,
                                      send_ranks, send_counts, send_ptrs);

    generic_unpack(map_incoming_buffers);
  }

  /**
   * Free the container.
   */
  inline void free() { this->ece.free(); }
};

} // namespace NESO::Particles::MeshHierarchyData

#endif
