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

  INT get_index(std::tuple<INT, INT, INT> vert);

  std::tuple<INT, INT, INT> to_standard_tuple(const INT linear_index);

  std::vector<std::tuple<INT, INT, INT>> get_coarse_tuples();

  std::vector<std::pair<std::tuple<INT, INT, INT>, std::tuple<INT, INT, INT>>>
  get_lines(std::tuple<INT, INT, INT> base);

  void write_inner(std::string filename, const bool fine);

public:
  /**
   *  Create new instance of the writer.
   *
   *  @param[in] mesh_hierarchy MeshHierarchy instance to use as source for
   *  cells.
   */
  VTKMeshHierarchyCellsWriter(std::shared_ptr<MeshHierarchy> mesh_hierarchy)
      : mesh_hierarchy(mesh_hierarchy), next_vert_index(0){};

  /**
   *  Add a cell to the list of cells to be written to the output file.
   *
   *  @param[in] linear_index Index of cell.
   */
  void push_back(const INT linear_index);

  /**
   *  Write the output vtk file.
   *
   *  @param[in] filename Filename to write output to. Should end in .vtk.
   */
  void write(std::string filename);

  /**
   * Write all fine cells to the output file.
   *
   * @param filename Filename to write fine cells to.
   */
  void write_all_fine(std::string filename);

  /**
   * Write all coarse cells to the output file.
   *
   * @param filename Filename to write fine cells to.
   */
  void write_all_coarse(std::string filename);
};

} // namespace NESO::Particles

#endif
