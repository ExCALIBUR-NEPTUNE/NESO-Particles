#include <neso_particles/mesh_hierarchy.hpp>

namespace NESO::Particles {

void MeshHierarchy::all_reduce_max_map() {
  MPICHK(MPI_Win_fence(0, this->map_win))
  if (comm_pair.rank_intra == 0) {
    int *tmp_map = (int *)std::malloc(this->ncells_global * sizeof(int));
    NESOASSERT(tmp_map != NULL, "malloc failed");

    for (int cellx = 0; cellx < this->ncells_global; cellx++) {
      tmp_map[cellx] = this->map[cellx];
    }

    MPICHK(MPI_Allreduce(tmp_map, this->map, this->ncells_global, MPI_INT,
                         MPI_MAX, comm_pair.comm_inter))

    std::free(tmp_map);
  }
  MPICHK(MPI_Win_fence(0, this->map_win))
};

MeshHierarchy::MeshHierarchy(MPI_Comm comm, const int ndim,
                             std::vector<int> dims, std::vector<double> origin,
                             const double extent, const int subdivision_order)
    : comm(comm), comm_pair(comm),
      global_move_communication(
          std::make_shared<GlobalMoveCommunication>(comm)),
      ndim(ndim), dims(dims), origin(origin),
      subdivision_order(subdivision_order), cell_width_coarse(extent),
      cell_width_fine(extent / ((double)std::pow(2, subdivision_order))),
      inverse_cell_width_coarse(1.0 / extent),
      inverse_cell_width_fine(((double)std::pow(2, subdivision_order)) /
                              extent),
      ncells_coarse(reduce_mul(ndim, dims)),
      ncells_fine(std::pow(std::pow(2, subdivision_order), ndim)) {
  NESOASSERT(ndim > 0, "ndim negative");
  NESOASSERT(dims.size() >= static_cast<std::size_t>(ndim),
             "vector of dims too small");
  NESOASSERT(origin.size() >= static_cast<std::size_t>(ndim),
             "vector of origin too small");
  for (int dimx = 0; dimx < ndim; dimx++) {
    NESOASSERT(dims[dimx] > 0, "Dim size is <= 0 in a direction.");
  }
  NESOASSERT(cell_width_coarse > 0.0, "Extent <= 0.0 passed");
  NESOASSERT(subdivision_order >= 0, "Negative subdivision order passed.");
  ncells_dim_fine = std::pow(2, subdivision_order);

  NESOASSERT(ncells_fine > 0, "Number of fine ncells does not make sense.");

  ncells_global = ncells_coarse * ncells_fine;

  // allocate space for the map stored on each shared memory region
  const MPI_Aint num_alloc =
      (comm_pair.rank_intra == 0) ? ncells_global * sizeof(int) : 0;
  MPICHK(MPI_Win_allocate_shared(num_alloc, sizeof(int), MPI_INFO_NULL,
                                 comm_pair.comm_intra, (void *)&this->map_base,
                                 &this->map_win))

  map_allocated = true;
  // map_base points to this ranks region - we want the base to the entire
  // allocated block in map.
  MPI_Aint win_size_tmp;
  int disp_unit_tmp;
  MPICHK(MPI_Win_shared_query(this->map_win, 0, &win_size_tmp, &disp_unit_tmp,
                              (void *)&this->map))
  NESOASSERT(static_cast<MPI_Aint>(ncells_global * sizeof(int)) == win_size_tmp,
             "Pointer to incorrect size.");
}

void MeshHierarchy::free() {
  if (this->map_allocated) {
    MPICHK(MPI_Win_free(&this->map_win))
    this->map = NULL;
    this->map_base = NULL;
    this->map_allocated = false;
  }
  this->comm_pair.free();
  this->global_move_communication->free();
}

INT MeshHierarchy::tuple_to_linear_global(INT *index_tuple) {
  INT index_coarse = tuple_to_linear_coarse(index_tuple);
  INT index_fine = tuple_to_linear_fine(&index_tuple[ndim]);
  INT index = index_coarse * ncells_fine + index_fine;
  return index;
};

INT MeshHierarchy::tuple_to_linear_coarse(INT *index_tuple) {
  INT index = index_tuple[ndim - 1];
  for (int dimx = ndim - 2; dimx >= 0; dimx--) {
    index *= dims[dimx];
    index += index_tuple[dimx];
  }
  return index;
}

INT MeshHierarchy::tuple_to_linear_fine(INT *index_tuple) {
  INT index = index_tuple[ndim - 1];
  for (int dimx = ndim - 2; dimx >= 0; dimx--) {
    index *= ncells_dim_fine;
    index += index_tuple[dimx];
  }
  return index;
}

void MeshHierarchy::linear_to_tuple_global(INT linear, INT *index) {
  auto pq = std::div((long long)linear, (long long)ncells_fine);
  linear_to_tuple_coarse(pq.quot, index);
  linear_to_tuple_fine(pq.rem, index + ndim);
}

void MeshHierarchy::linear_to_tuple_coarse(INT linear, INT *index) {
  for (int dimx = 0; dimx < ndim; dimx++) {
    auto pq = std::div((long long)linear, (long long)dims[dimx]);
    index[dimx] = pq.rem;
    linear = pq.quot;
  }
}

void MeshHierarchy::linear_to_tuple_fine(INT linear, INT *index) {
  for (int dimx = 0; dimx < ndim; dimx++) {
    auto pq = std::div((long long)linear, (long long)ncells_dim_fine);
    index[dimx] = pq.rem;
    linear = pq.quot;
  }
}

void MeshHierarchy::claim_initialise() {
  NESOASSERT(claim_stack.empty(), "Claim stack is not empty.");

  // initialise the memory in the shared memory window
  MPICHK(MPI_Win_fence(0, this->map_win))
  INT start, end;
  get_decomp_1d((INT)this->comm_pair.size_intra, ncells_global,
                (INT)this->comm_pair.rank_intra, &start, &end);
  for (INT cellx = start; cellx < end; cellx++) {
    map[cellx] = mask;
  }
  MPICHK(MPI_Win_fence(0, this->map_win))

  MPICHK(MPI_Barrier(this->comm_pair.comm_intra))
  MPICHK(MPI_Win_fence(0, this->map_win))
}

void MeshHierarchy::claim_cell(const INT index, int weight) {
  NESOASSERT((weight != mask) && (weight > 0) && (weight < (-1 * (mask + 1))),
             "Invalid weight");
  weight += mask;
  std::pair<INT, int> claim{index, weight};
  claim_stack.push(claim);

  MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, map_win);
  int result;
  MPICHK(MPI_Get_accumulate(&weight, 1, MPI_INT, &result, 1, MPI_INT, 0,
                            (MPI_Aint)index, 1, MPI_INT, MPI_MAX, map_win));

  MPI_Win_unlock(0, map_win);
}

void MeshHierarchy::claim_finalise() {

  MPICHK(MPI_Win_fence(0, this->map_win));
  MPICHK(MPI_Barrier(this->comm_pair.comm_intra));

  // reduce accross all shared regions (inter node)
  all_reduce_max_map();

  NESOASSERT(claim_stack_binned.empty(), "Claim stack is not empty.");

  while (!claim_stack.empty()) {
    auto claim = claim_stack.top();

    // if the weight of this claim is in the cell then this rank is one of
    // the ranks with the maximum weight.
    if (claim.second >= map[claim.first]) {
      claim_stack_binned.push(claim);
    }
    claim_stack.pop();
  }

  MPICHK(MPI_Barrier(this->comm_pair.comm_intra));

  // for each rank with the maximum claimed weight reduce the ranks
  const int rank = comm_pair.rank_parent;

  MPICHK(MPI_Win_fence(0, this->map_win));

  while (!claim_stack_binned.empty()) {
    auto claim = claim_stack_binned.top();
    INT index = claim.first;
    int result;

    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, map_win);
    MPICHK(MPI_Get_accumulate(&rank, 1, MPI_INT, &result, 1, MPI_INT, 0,
                              (MPI_Aint)index, 1, MPI_INT, MPI_MAX, map_win));
    MPI_Win_unlock(0, map_win);
    claim_stack_binned.pop();
  }

  MPICHK(MPI_Win_fence(0, this->map_win));

  MPICHK(MPI_Barrier(this->comm_pair.comm_intra));
  // reduce accross all shared regions (inter node)
  all_reduce_max_map();

  // this->map should now contain either values of "mask" for cells that were
  // not claimed and the rank of claimed cells.
  map_created = true;
}

int MeshHierarchy::get_owner(INT index) {
  NESOASSERT(map_created, "map is not created");
  return map[index];
}

void MeshHierarchy::get_owners(const int nqueries, INT *indices, int *ranks) {
  for (int qx = 0; qx < nqueries; qx++) {
    const INT linear_index =
        this->tuple_to_linear_global(indices + (qx * this->ndim * 2));
    const int rank = get_owner(linear_index);
    ranks[qx] = rank;
    if (Debug::enabled(Debug::MOVEMENT_LEVEL)) {
      nprint("MeshHierarchy::get_owners:", "qx:", qx,
             "linear_index:", linear_index, "rank:", rank);
    }
  }
}

void get_neighbour_mh_cells(std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                            const INT cell, const INT offset, const bool pbc,
                            std::set<INT> &output_cells) {
  const int ndim = mesh_hierarchy->ndim;
  const INT ncells_dim_fine = mesh_hierarchy->ncells_dim_fine;

  INT offset_starts[3] = {0, 0, 0};
  INT offset_ends[3] = {1, 1, 1};
  for (int dimx = 0; dimx < ndim; dimx++) {
    offset_starts[dimx] = -offset;
    offset_ends[dimx] = offset + 1;
  }

  INT cell_counts[3] = {1, 1, 1};
  for (int dimx = 0; dimx < ndim; dimx++) {
    cell_counts[dimx] = mesh_hierarchy->dims[dimx] * ncells_dim_fine;
  }

  INT global_tuple_mh[6] = {0, 0, 0, 0, 0, 0};
  INT global_tuple[3] = {0, 0, 0};
  mesh_hierarchy->linear_to_tuple_global(cell, global_tuple_mh);
  // convert the mesh hierary tuple format into a more standard tuple format
  for (int dimx = 0; dimx < ndim; dimx++) {
    const INT cart_index_dim =
        global_tuple_mh[dimx] * ncells_dim_fine + global_tuple_mh[dimx + ndim];
    global_tuple[dimx] = cart_index_dim;
  }

  // loop over the offsets
  INT ox[3];
  for (ox[2] = offset_starts[2]; ox[2] < offset_ends[2]; ox[2]++) {
    for (ox[1] = offset_starts[1]; ox[1] < offset_ends[1]; ox[1]++) {
      for (ox[0] = offset_starts[0]; ox[0] < offset_ends[0]; ox[0]++) {
        // compute the cell from the offset

        bool valid = true;
        for (int dimx = 0; dimx < ndim; dimx++) {
          const INT offset_dim_linear_unmapped = global_tuple[dimx] + ox[dimx];
          if (!pbc) {
            valid = valid && ((offset_dim_linear_unmapped >= 0) &&
                              (offset_dim_linear_unmapped < cell_counts[dimx]));
          }
          const INT offset_dim_linear =
              (offset_dim_linear_unmapped + cell_counts[dimx]) %
              cell_counts[dimx];

          // convert back to a mesh hierarchy tuple index
          auto pq = std::div((long long)offset_dim_linear,
                             (long long)ncells_dim_fine);
          global_tuple_mh[dimx] = pq.quot;
          global_tuple_mh[dimx + ndim] = pq.rem;
        }
        const INT offset_linear =
            mesh_hierarchy->tuple_to_linear_global(global_tuple_mh);

        NESOASSERT(offset_linear < mesh_hierarchy->ncells_global,
                   "bad offset index - too high");
        NESOASSERT(-1 < offset_linear, "bad offset index - too low");

        // if this rank owns this cell then there is nothing to do
        if (valid && (offset_linear != cell)) {
          output_cells.insert(offset_linear);
        }
      }
    }
  }
}

void get_neighbour_mh_cells(std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                            const std::vector<INT> &cells,
                            const std::vector<INT> &cells_extra,
                            const INT offset, const bool pbc,
                            std::vector<INT> &output_cells) {

  output_cells.clear();
  std::set<INT> output_cells_set;
  output_cells_set.clear();
  for (INT cx : cells) {
    output_cells_set.insert(cx);
  }
  for (INT cx : cells_extra) {
    output_cells_set.insert(cx);
  }
  for (INT cx : cells) {
    get_neighbour_mh_cells(mesh_hierarchy, cx, offset, pbc, output_cells_set);
  }
  output_cells.reserve(output_cells_set.size());
  for (const INT &cx : output_cells_set) {
    output_cells.push_back(cx);
  }
}

} // namespace NESO::Particles
