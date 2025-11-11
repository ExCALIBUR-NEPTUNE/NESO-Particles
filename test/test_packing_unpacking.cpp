#include "include/test_particle_sub_group.hpp"

TEST(PackingUnpacking, unpacking) {
  const int npart_cell = 211;
  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, 2);

  auto particle_group_pointer_map = std::make_shared<ParticleGroupPointerMap>(
      sycl_target, &A->particle_dats_real, &A->particle_dats_int);
  auto unpacker = std::make_shared<ParticleUnpacker>(sycl_target);

  const int num_remote_recv_ranks = 7;
  BufferHost<int> h_recv_rank_npart(sycl_target, num_remote_recv_ranks);

  std::size_t num_particles = 0;
  for (int rx = 0; rx < num_remote_recv_ranks; rx++) {
    h_recv_rank_npart.ptr[rx] = rx * rx;
    num_particles += rx * rx;
  }

  unpacker->reset(num_remote_recv_ranks, h_recv_rank_npart,
                  particle_group_pointer_map);

  char **recv_pointers = unpacker->get_recv_pointers(num_remote_recv_ranks);

  const std::size_t num_bytes_per_particle =
      particle_group_pointer_map->get_num_bytes_per_particle();
  std::vector<char> h_to_unpack(num_bytes_per_particle * num_particles);

  int index = 0;
  char *buffer = h_to_unpack.data();
  for (int rx = 0; rx < num_remote_recv_ranks; rx++) {
    const int num_particles_rank = rx * rx;

    for (int px = 0; px < num_particles_rank; px++) {
      for (auto &[sym, dat] : A->particle_dats_real) {
        const int ncomp = dat->ncomp;
        for (int cx = 0; cx < ncomp; cx++) {
          REAL v = index++;
          std::memcpy(buffer, &v, sizeof(REAL));
          buffer += sizeof(REAL);
        }
      }
      for (auto &[sym, dat] : A->particle_dats_int) {
        const int ncomp = dat->ncomp;
        for (int cx = 0; cx < ncomp; cx++) {
          INT v = index++;
          std::memcpy(buffer, &v, sizeof(INT));
          buffer += sizeof(INT);
        }
      }
    }
  }

  ASSERT_EQ(buffer,
            h_to_unpack.data() + num_bytes_per_particle * num_particles);

  buffer = h_to_unpack.data();
  for (int rx = 0; rx < num_remote_recv_ranks; rx++) {
    const int num_particles_rank = rx * rx;
    if (num_particles_rank) {
      sycl_target->queue
          .memcpy(recv_pointers[rx], buffer,
                  num_particles_rank * num_bytes_per_particle)
          .wait_and_throw();
    }
    buffer += num_particles_rank * num_bytes_per_particle;
  }

  A->clear();
  unpacker->unpack(particle_group_pointer_map);

  index = 0;
  int qx = 0;
  for (int rx = 0; rx < num_remote_recv_ranks; rx++) {
    const int num_particles_rank = rx * rx;

    for (int px = 0; px < num_particles_rank; px++) {
      for (auto &[sym, dat] : A->particle_dats_real) {
        const int ncomp = dat->ncomp;
        for (int cx = 0; cx < ncomp; cx++) {
          REAL v = index++;
          ASSERT_EQ(v, dat->cell_dat.get_value(0, qx, cx));
        }
      }
      for (auto &[sym, dat] : A->particle_dats_int) {
        const int ncomp = dat->ncomp;
        for (int cx = 0; cx < ncomp; cx++) {
          INT v = index++;
          ASSERT_EQ(v, dat->cell_dat.get_value(0, qx, cx));
        }
      }
      qx++;
    }
  }

  A->free();
  sycl_target->free();
  A->domain->mesh->free();
}

TEST(PackingUnpacking, packing) {
  const int npart_cell = 79;
  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, 2);

  auto particle_group_pointer_map = std::make_shared<ParticleGroupPointerMap>(
      sycl_target, &A->particle_dats_real, &A->particle_dats_int);
  auto packer = std::make_shared<ParticlePacker>(sycl_target);

  packer->reset();

  const int num_remote_ranks = sycl_target->comm_pair.size_parent;
  int send_rank_index = 0;

  std::map<int, std::vector<std::pair<int, int>>> map_rank_cell_layers;

  std::vector<int> h_pack_cells;
  std::vector<int> h_pack_layers_src;
  std::vector<int> h_pack_layers_dst;

  int num_particles_leaving = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto ID = A->get_cell(Sym<INT>("ID"), cellx);

    auto mpi_rank_dat = A->mpi_rank_dat;
    const int nrow = ID->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      const int send_rank = (send_rank_index++) % num_remote_ranks;
      h_pack_cells.push_back(cellx);
      h_pack_layers_src.push_back(rowx);
      h_pack_layers_dst.push_back(map_rank_cell_layers[send_rank].size());
      map_rank_cell_layers[send_rank].push_back({cellx, rowx});
      mpi_rank_dat->cell_dat.set_value(cellx, rowx, 0, send_rank);
      num_particles_leaving++;
    }
  }

  BufferHost<int> h_send_rank_npart(sycl_target, num_remote_ranks);

  std::vector<int> t_iota(num_remote_ranks);
  std::iota(t_iota.begin(), t_iota.end(), 0);
  BufferDeviceHost<int> dh_send_rank_map(sycl_target, t_iota);
  BufferDevice d_pack_cells(sycl_target, h_pack_cells);
  BufferDevice d_pack_layers_src(sycl_target, h_pack_layers_src);
  BufferDevice d_pack_layers_dst(sycl_target, h_pack_layers_dst);

  int index = 0;
  for (auto &m : map_rank_cell_layers) {
    ASSERT_EQ(index, m.first);
    h_send_rank_npart.ptr[index] = static_cast<int>(m.second.size());
    index++;
  }
  dh_send_rank_map.host_to_device();

  EventStack event_stack;
  packer->pack(num_remote_ranks, h_send_rank_npart, dh_send_rank_map,
               num_particles_leaving, d_pack_cells, d_pack_layers_src,
               d_pack_layers_dst, particle_group_pointer_map, event_stack);
  event_stack.wait();

  char **all_buffers =
      packer->get_packed_pointers(num_remote_ranks, h_send_rank_npart.ptr);

  const std::size_t num_bytes_per_particle =
      particle_group_pointer_map->get_num_bytes_per_particle();

  index = 0;
  for (auto &[rank, cells_layers] : map_rank_cell_layers) {

    const int num_particles = h_send_rank_npart.ptr[index];
    if (num_particles) {
      std::vector<char> h_buffer(num_particles * num_bytes_per_particle);
      sycl_target->queue
          .memcpy(h_buffer.data(), all_buffers[index],
                  num_particles * num_bytes_per_particle)
          .wait_and_throw();

      char *buffer = h_buffer.data();
      for (auto &cell_layer : cells_layers) {
        const int cell = cell_layer.first;
        const int layer = cell_layer.second;
        for (auto &[sym, dat] : A->particle_dats_real) {
          const int ncomp = dat->ncomp;
          for (int cx = 0; cx < ncomp; cx++) {
            REAL v = 0;
            std::memcpy(&v, buffer, sizeof(REAL));
            buffer += sizeof(REAL);
            ASSERT_EQ(v, dat->cell_dat.get_value(cell, layer, cx));
          }
        }
        for (auto &[sym, dat] : A->particle_dats_int) {
          const int ncomp = dat->ncomp;
          for (int cx = 0; cx < ncomp; cx++) {
            INT v = 0;
            std::memcpy(&v, buffer, sizeof(INT));
            buffer += sizeof(INT);
            ASSERT_EQ(v, dat->cell_dat.get_value(cell, layer, cx));
          }
        }
      }

      ASSERT_EQ(buffer,
                h_buffer.data() + cells_layers.size() * num_bytes_per_particle);
    }
    index++;
  }

  A->free();
  sycl_target->free();
  A->domain->mesh->free();
}
