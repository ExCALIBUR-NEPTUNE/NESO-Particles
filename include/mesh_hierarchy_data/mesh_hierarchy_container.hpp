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

struct MHCellBuffer : public SerialInterface {
  INT cell;
  std::vector<std::byte> buffer;

  MHCellBuffer() = default;
  MHCellBuffer(INT cell, std::vector<std::byte> &buffer) : cell(cell), buffer(buffer) {};

  virtual inline std::size_t get_num_bytes() const override {
    return sizeof(INT) + sizeof(std::size_t) + this->buffer.size();
  }
  virtual inline void
  serialise([[maybe_unused]] std::byte *buffer,
            [[maybe_unused]] const std::size_t num_bytes) const override {
    const std::byte * check_end = buffer + num_bytes;
    std::memcpy(buffer, &this->cell, sizeof(INT));
    buffer += sizeof(INT);
    const std::size_t buffer_size = this->buffer.size();
    nprint("pushing buffer_size:", buffer_size);
    std::memcpy(buffer, &buffer_size, sizeof(std::size_t));
    buffer += sizeof(size_t);
    std::memcpy(buffer, this->buffer.data(), buffer_size);
    NESOASSERT(
    buffer + buffer_size == check_end,
    "Packing under/overflow occured.");
  }
  virtual inline void
  deserialise([[maybe_unused]] const std::byte *buffer,
              [[maybe_unused]] const std::size_t num_bytes) override {
    const std::byte * check_end = buffer + num_bytes;
    std::memcpy(&this->cell, buffer, sizeof(INT));
    buffer += sizeof(INT);
    std::size_t buffer_size = std::numeric_limits<std::size_t>::max();
    std::memcpy(&buffer_size, buffer, sizeof(std::size_t));
    buffer += sizeof(std::size_t);
    NESOASSERT(buffer_size < std::numeric_limits<std::size_t>::max(),
      "Unexpected buffer size.");
    this->buffer.resize(buffer_size);
    nprint("poping  buffer_size:", buffer_size);
    std::memcpy(this->buffer.data(), buffer, buffer_size);
    NESOASSERT(
    buffer + buffer_size == check_end,
    "Packing under/overflow occured.");
  }
};

/**
 * TODO
 */
template <typename T> class MeshHierarchyContainer {
protected:
  MPI_Comm comm;
  CommunicationEdgesCounter ece;
  std::map<int, SerialContainer<T>> map_cell_buffers;

public:
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;

  /**
   * TODO
   * collective
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
      MHCellBuffer mhcbt(cell, cell_contents.buffer);

      std::vector<MHCellBuffer> tmp_cell;
      tmp_cell.emplace_back(cell, cell_contents.buffer);
      nprint("T:", tmp_cell.at(0).buffer.size(), tmp_cell.at(0).buffer.data(), tmp_cell.at(0).buffer.data() + tmp_cell.at(0).buffer.size());
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
    this->ece.get_remote_ranks(
      recv_ranks,
      recv_counts,
      send_ranks,
      send_counts
    );
    const int num_send_ranks = send_ranks.size();
    send_ptrs.reserve(num_send_ranks);

    // Allocate space for the incoming serialised data per rank
    std::map<int, SerialContainer<MHCellBuffer>> map_incoming_buffers;
    for(int rankx=0 ; rankx<num_send_ranks ; rankx++){
      const int rank = send_ranks.at(rankx);
      const std::size_t num_bytes = static_cast<std::size_t>(send_counts.at(rankx));
      map_incoming_buffers[rank] = SerialContainer<MHCellBuffer>(num_bytes);
      send_ptrs.push_back(map_incoming_buffers.at(rank).buffer.data());
      nprint(rankx, rank, map_incoming_buffers.at(rank).buffer.size());
    }
    
    nprint("----------------------");

    // exchange serialised data
    this->ece.exchange_send_recv_data(
      recv_ranks,
      recv_counts,
      recv_ptrs,
      send_ranks,
      send_counts,
      send_ptrs
    );
    
    std::vector<MHCellBuffer> tmp;
    // unpack the rank wise data into cell wise data
    for(auto &item : map_incoming_buffers){
      auto &buffer = item.second;
      buffer.get(tmp);
      for(auto & tx : tmp){
        const INT cell = tx.cell;
        auto &inner_buffer = tx.buffer;
        SerialContainer<T> tmp_serial(inner_buffer.size());
        std::memcpy(tmp_serial.buffer.data(), inner_buffer.data(), inner_buffer.size());
        nprint(inner_buffer.size());
        map_cell_buffers[cell].append(tmp_serial);
      }
    }
  }

  /**
   * TODO
   */
  inline void free() { this->ece.free(); }
};

} // namespace NESO::Particles::MeshHierarchyData

#endif
