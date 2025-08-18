#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/common_impl.hpp>
#include <neso_particles/external_interfaces/petsc/project_evaluate/dmplex_project_evaluate_barycentric.hpp>

namespace NESO::Particles::PetscInterface {

std::vector<VTK::UnstructuredCell>
DMPlexProjectEvaluateBarycentric::get_vtk_data() {
  const int cell_count = this->mesh->get_cell_count();
  std::vector<VTK::UnstructuredCell> data(cell_count);
  std::vector<std::vector<REAL>> vertices;
  const int ndim = mesh->get_ndim();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    vertices.clear();
    mesh->dmh->get_cell_vertices(cellx, vertices);
    auto values = this->cdc_project->get_cell(cellx);
    auto inverse_volumes = this->cdc_volumes->get_value(cellx, 0, 0);
    const int num_vertices = vertices.size();
    data.at(cellx).num_points = num_vertices;
    data.at(cellx).cell_type = num_vertices == 3 ? VTK::CellType::triangle
                                                 : VTK::CellType::quadrilateral;
    data.at(cellx).points.reserve(num_vertices * 3);
    data.at(cellx).point_data["value"].reserve(num_vertices);
    for (int vx = 0; vx < num_vertices; vx++) {
      for (int dx = 0; dx < ndim; dx++) {
        data.at(cellx).points.push_back(vertices.at(vx).at(dx));
      }
      for (int dx = ndim; dx < 3; dx++) {
        data.at(cellx).points.push_back(0.0);
      }
      data.at(cellx).point_data["value"].push_back(values->at(0, vx) *
                                                   inverse_volumes);
    }
  }
  return data;
}

DMPlexProjectEvaluateBarycentric::DMPlexProjectEvaluateBarycentric(
    ExternalCommon::QuadraturePointMapperSharedPtr qpm,
    std::string function_space, int polynomial_order, const bool testing)
    : qpm(qpm), function_space(function_space),
      polynomial_order(polynomial_order) {

  std::map<std::string, std::pair<int, int>> map_allowed;
  map_allowed["Barycentric"] = {1, 1};

  NESOASSERT(this->qpm != nullptr, "QuadraturePointMapper is nullptr");
  NESOASSERT(map_allowed.count(function_space),
             "Only function space: " + function_space + " not recognised.");
  const int p_min = map_allowed.at(function_space).first;
  const int p_max = map_allowed.at(function_space).second;
  NESOASSERT(((p_min <= polynomial_order) && (polynomial_order <= p_max)),
             "Polynomial order " + std::to_string(polynomial_order) +
                 " outside of acceptable range [" + std::to_string(p_min) +
                 ", " + std::to_string(p_max) + "] for function space " +
                 function_space + ".");

  this->mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
      this->qpm->domain->mesh);
  NESOASSERT(this->mesh != nullptr,
             "Mesh is not descendent from PetscInterface::DMPlexInterface");
  NESOASSERT(this->mesh->get_ndim() == 2, "Only implemented for 2D domains.");

  const int cell_count = this->qpm->domain->mesh->get_cell_count();
  this->cdc_project = std::make_shared<CellDatConst<REAL>>(
      this->qpm->sycl_target, cell_count, 1, 4);
  this->cdc_num_vertices = std::make_shared<CellDatConst<int>>(
      this->qpm->sycl_target, cell_count, 1, 1);
  this->cdc_vertices = std::make_shared<CellDatConst<REAL>>(
      this->qpm->sycl_target, cell_count, 4, 2);
  this->cdc_matrices = nullptr;

  // For each cell record the verticies.
  std::vector<std::vector<REAL>> vertices;
  for (int cx = 0; cx < cell_count; cx++) {
    this->mesh->dmh->get_cell_vertices(cx, vertices);
    const int num_verts = vertices.size();
    NESOASSERT(testing || ((2 < num_verts) && (num_verts < 5)),
               "Unexpected number of vertices (expected 3 or 4).");
    this->cdc_num_vertices->set_value(cx, 0, 0, num_verts);
    for (int vx = 0; vx < num_verts; vx++) {
      for (int dx = 0; dx < 2; dx++) {
        this->cdc_vertices->set_value(cx, vx, dx, vertices.at(vx).at(dx));
      }
    }
  }
  this->cdc_volumes = std::make_shared<CellDatConst<REAL>>(
      this->qpm->sycl_target, cell_count, 1, 1);

  // For each cell record the volume.
  for (int cx = 0; cx < cell_count; cx++) {
    const auto volume = this->mesh->dmh->get_cell_volume(cx);
    this->cdc_volumes->set_value(
        cx, 0, 0, this->cdc_num_vertices->get_value(cx, 0, 0) / volume);
  }

  this->setup_matrices();
}

void DMPlexProjectEvaluateBarycentric::project(
    ParticleGroupSharedPtr particle_group, Sym<REAL> sym) {
  this->project_inner(particle_group, sym);
}

void DMPlexProjectEvaluateBarycentric::project(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym) {
  this->project_inner(particle_sub_group, sym);
}

void DMPlexProjectEvaluateBarycentric::evaluate(
    ParticleGroupSharedPtr particle_group, Sym<REAL> sym) {
  this->evaluate_inner(particle_group, sym);
}

void DMPlexProjectEvaluateBarycentric::evaluate(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym) {
  this->evaluate_inner(particle_sub_group, sym);
}

} // namespace NESO::Particles::PetscInterface
#endif
