#ifndef _NESO_PARTICLES_UTILITY_MESH_HIERARCHY_PLOTTING
#define _NESO_PARTICLES_UTILITY_MESH_HIERARCHY_PLOTTING

#include "mesh_hierarchy.hpp"
#include "typedefs.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace NESO::Particles {

/**
 * Class to write MeshHierarchy cells to a vtk file as a collection of vertices
 * and edges for visualisation in Paraview.
 */
class VTKMeshHierarchyCellsWriter {
protected:
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;
  std::vector<INT> cells;
  std::map<std::tuple<INT, INT, INT>, INT> verts_to_index;
  std::map<INT, std::tuple<INT, INT, INT>> index_to_verts;

  INT next_vert_index;

  inline INT get_index(std::tuple<INT, INT, INT> vert) {

    if (verts_to_index.count(vert)) {
      return verts_to_index[vert];
    } else {
      const INT tmp_vert_index = next_vert_index++;
      verts_to_index[vert] = tmp_vert_index;
      index_to_verts[tmp_vert_index] = vert;
      return tmp_vert_index;
    }
  }

  inline std::tuple<INT, INT, INT> to_standard_tuple(const INT linear_index) {
    const int ndim = this->mesh_hierarchy->ndim;
    INT tuple_cell[6];

    INT stuple[3] = {0, 0, 0};
    this->mesh_hierarchy->linear_to_tuple_global(linear_index, tuple_cell);
    const INT num_cells_fine = this->mesh_hierarchy->ncells_dim_fine;
    for (int dimx = 0; dimx < ndim; dimx++) {
      stuple[dimx] =
          tuple_cell[dimx] * num_cells_fine + tuple_cell[dimx + ndim];
    }
    return {stuple[0], stuple[1], stuple[2]};
  }

  inline std::vector<std::tuple<INT, INT, INT>> get_coarse_tuples() {

    std::array<INT, 3> ends = {1, 1, 1};

    const int ndim = this->mesh_hierarchy->ndim;
    ends.at(0) = this->mesh_hierarchy->dims.at(0);
    if (ndim > 1) {
      ends.at(1) = this->mesh_hierarchy->dims.at(1);
    }
    if (ndim > 2) {
      ends.at(2) = this->mesh_hierarchy->dims.at(2);
    }

    std::vector<std::tuple<INT, INT, INT>> cells;
    cells.reserve(ends.at(0) * ends.at(1) * ends.at(2));

    std::array<INT, 3> ix;
    for (ix[2] = 0; ix[2] < ends[2]; ix[2]++) {
      for (ix[1] = 0; ix[1] < ends[1]; ix[1]++) {
        for (ix[0] = 0; ix[0] < ends[0]; ix[0]++) {
          cells.push_back({ix[0], ix[1], ix[2]});
        }
      }
    }

    return cells;
  }

  inline std::vector<
      std::pair<std::tuple<INT, INT, INT>, std::tuple<INT, INT, INT>>>
  get_lines(std::tuple<INT, INT, INT> base) {
    std::vector<std::pair<std::tuple<INT, INT, INT>, std::tuple<INT, INT, INT>>>
        lines;

    const int ndim = this->mesh_hierarchy->ndim;
    const INT bx = std::get<0>(base);
    const INT by = std::get<1>(base);
    const INT bz = std::get<2>(base);

    // case for ndim == 1
    lines.push_back({{bx, by, bz}, {bx + 1, by, bz}});
    if (ndim > 1) {
      lines.push_back({{bx, by + 1, bz}, {bx + 1, by + 1, bz}});
      lines.push_back({{bx, by, bz}, {bx, by + 1, bz}});
      lines.push_back({{bx + 1, by, bz}, {bx + 1, by + 1, bz}});
    }

    if (ndim > 2) {
      lines.push_back({{bx, by, bz + 1}, {bx + 1, by, bz + 1}});
      lines.push_back({{bx, by + 1, bz + 1}, {bx + 1, by + 1, bz + 1}});
      lines.push_back({{bx, by, bz + 1}, {bx, by + 1, bz + 1}});
      lines.push_back({{bx + 1, by, bz + 1}, {bx + 1, by + 1, bz + 1}});

      lines.push_back({{bx, by, bz}, {bx, by, bz + 1}});
      lines.push_back({{bx + 1, by, bz}, {bx + 1, by, bz + 1}});
      lines.push_back({{bx + 1, by + 1, bz}, {bx + 1, by + 1, bz + 1}});
      lines.push_back({{bx, by + 1, bz}, {bx, by + 1, bz + 1}});
    }

    return lines;
  }

  inline void write_inner(std::string filename, const bool fine) {
    this->next_vert_index = 0;
    this->verts_to_index.clear();
    this->index_to_verts.clear();
    const int ndim = this->mesh_hierarchy->ndim;
    std::vector<std::pair<INT, INT>> edges;

    auto lambda_push_edges = [&](auto lines) {
      for (auto linex : lines) {
        const INT index_start = this->get_index(linex.first);
        const INT index_end = this->get_index(linex.second);
        edges.push_back({index_start, index_end});
      }
    };

    if (fine) {
      for (const INT linear_index : this->cells) {
        auto base_corner = to_standard_tuple(linear_index);
        auto lines = this->get_lines(base_corner);
        lambda_push_edges(lines);
      }
    } else {
      auto coarse_cells = this->get_coarse_tuples();
      for (auto cx : coarse_cells) {
        auto lines = this->get_lines(cx);
        lambda_push_edges(lines);
      }
    }

    std::ofstream vtk_file;
    vtk_file.open(filename);

    vtk_file << "# vtk DataFile Version 2.0\n";
    vtk_file << "NESO-Particles Mesh Hierarchy\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET UNSTRUCTURED_GRID\n\n";
    vtk_file << "POINTS " << this->index_to_verts.size() << " float\n";

    const double cell_width = fine ? this->mesh_hierarchy->cell_width_fine
                                   : this->mesh_hierarchy->cell_width_coarse;

    auto origin_mh = this->mesh_hierarchy->origin;
    double origin[3] = {0.0, 0.0, 0.0};
    for (int dimx = 0; dimx < ndim; dimx++) {
      origin[dimx] = origin_mh[dimx];
    }

    for (INT vx = 0; vx < this->next_vert_index; vx++) {
      auto vertex = this->index_to_verts[vx];
      const double x = ((double)std::get<0>(vertex)) * cell_width + origin[0];
      const double y = ((double)std::get<1>(vertex)) * cell_width + origin[1];
      const double z = ((double)std::get<2>(vertex)) * cell_width + origin[2];
      vtk_file << x << " " << y << " " << z << " \n";
    }
    vtk_file << "\n";

    std::vector<INT> edge_ints;
    for (auto &edge : edges) {
      edge_ints.push_back(2);
      edge_ints.push_back(edge.first);
      edge_ints.push_back(edge.second);
    }

    const int num_edges = edges.size();
    vtk_file << "CELLS " << num_edges << " " << edge_ints.size() << "\n";
    for (const int ix : edge_ints) {
      vtk_file << ix << " ";
    }
    vtk_file << "\n";
    vtk_file << "\n";
    vtk_file << "CELL_TYPES " << num_edges << "\n";
    for (int ix = 0; ix < num_edges; ix++) {
      vtk_file << 3 << " ";
    }

    vtk_file.close();
  }

public:
  /**
   *  Create new instance of the writer.
   *
   *  @param[in] mesh_hierarchy MeshHierarchy instance to use as source for
   *  cells.
   */
  VTKMeshHierarchyCellsWriter(std::shared_ptr<MeshHierarchy> mesh_hierarchy)
      : mesh_hierarchy(mesh_hierarchy), next_vert_index(0) {};

  /**
   *  Add a cell to the list of cells to be written to the output file.
   *
   *  @param[in] linear_index Index of cell.
   */
  inline void push_back(const INT linear_index) {
    this->cells.push_back(linear_index);
  }

  /**
   *  Write the output vtk file.
   *
   *  @param[in] filename Filename to write output to. Should end in .vtk.
   */
  inline void write(std::string filename) { write_inner(filename, true); }

  /**
   * Write all fine cells to the output file.
   *
   * @param filename Filename to write fine cells to.
   */
  inline void write_all_fine(std::string filename) {
    const int rank = this->mesh_hierarchy->comm_pair.rank_parent;
    if (rank == 0) {
      const INT ncells_global = this->mesh_hierarchy->ncells_global;
      for (INT cx = 0; cx < ncells_global; cx++) {
        this->push_back(cx);
      }
      this->write(filename);
    }
  }

  /**
   * Write all coarse cells to the output file.
   *
   * @param filename Filename to write fine cells to.
   */
  inline void write_all_coarse(std::string filename) {
    const int rank = this->mesh_hierarchy->comm_pair.rank_parent;
    if (rank == 0) {
      write_inner(filename, false);
    }
  }
};

} // namespace NESO::Particles

#endif
