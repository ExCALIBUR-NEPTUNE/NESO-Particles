#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>

using namespace NESO::Particles;

class PETSC_NDIM : public testing::TestWithParam<int> {};
TEST_P(PETSC_NDIM, init) {
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PetscInt ndim = GetParam();

  const int mesh_size = (ndim == 2) ? 32 : 16;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               /* lower */ NULL,
                               /* upper */ NULL,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));
  PetscInterface::generic_distribute(&dm);
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  PetscInt start, end;
  PETSCCHK(DMPlexGetHeightStratum(dm, 0, &start, &end));
  ASSERT_EQ(mesh->get_cell_count(), end - start);
  ASSERT_EQ(mesh->get_ndim(), ndim);

  auto mh = mesh->get_mesh_hierarchy();

  ASSERT_EQ(mh->dims[0], 1);
  ASSERT_EQ(mh->dims[1], 1);
  if (ndim > 2) {
    ASSERT_EQ(mh->dims[2], 1);
  }
  ASSERT_EQ(mh->ncells_dim_fine, mesh_size);

  // The mesh hierarchy ownership should match the mesh ownership as it is a
  // cartesian mesh.
  int rank;
  MPICHK(MPI_Comm_rank(mesh->get_comm(), &rank));

  const int cell_count = mesh->get_cell_count();
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  auto mh_mapper = std::make_shared<MeshHierarchyMapper>(sycl_target, mh);
  auto mapper = mh_mapper->get_host_mapper();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    // Test the bounding box contains the average of the vertices
    std::vector<REAL> cell_average(ndim);
    mesh->dmh->get_cell_vertex_average(cellx, cell_average);
    auto bb = mesh->dmh->get_cell_bounding_box(cellx);
    ASSERT_TRUE(bb->contains_point(ndim, cell_average));

    // Check cell claimed the corresponding MH cell
    INT mh_tuple[6];
    mapper.map_to_tuple(cell_average.data(), mh_tuple);
    const INT mh_linear_index = mapper.tuple_to_linear_global(mh_tuple);
    const int owner = mh->get_owner(mh_linear_index);
    ASSERT_EQ(owner, rank);
  }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSC, create_dm) {

  PETSCCHK(PetscInitializeNoArguments());

  // start of mesh creation

  DM dm;

  /*
   * 8--3--7
   * |     |
   * 4  0  2
   * |     |
   * 5--1--6
   */

  PETSCCHK(DMCreate(MPI_COMM_WORLD, &dm));
  PETSCCHK(DMSetType(dm, DMPLEX));
  PETSCCHK(DMSetDimension(dm, 2));
  PETSCCHK(DMPlexSetChart(dm, 0, 9));

  PETSCCHK(DMPlexSetConeSize(dm, 0, 4));
  PETSCCHK(DMPlexSetConeSize(dm, 1, 2));
  PETSCCHK(DMPlexSetConeSize(dm, 2, 2));
  PETSCCHK(DMPlexSetConeSize(dm, 3, 2));
  PETSCCHK(DMPlexSetConeSize(dm, 4, 2));
  PETSCCHK(DMSetUp(dm));

  std::vector<PetscInt> pts;
  pts = {1, 2, 3, 4};
  PETSCCHK(DMPlexSetCone(dm, 0, pts.data()));
  pts = {5, 6};
  PETSCCHK(DMPlexSetCone(dm, 1, pts.data()));
  pts = {6, 7};
  PETSCCHK(DMPlexSetCone(dm, 2, pts.data()));
  pts = {7, 8};
  PETSCCHK(DMPlexSetCone(dm, 3, pts.data()));
  pts = {8, 5};
  PETSCCHK(DMPlexSetCone(dm, 4, pts.data()));
  PETSCCHK(DMPlexSymmetrize(dm));
  PETSCCHK(DMPlexStratify(dm));

  // create coordinates section for dm
  const PetscInt ndim = 2;
  PETSCCHK(DMSetCoordinateDim(dm, ndim));
  const PetscInt vertex_start = 5;
  const PetscInt vertex_end = vertex_start + 4;

  /*
  PetscSection coord_section;
  PETSCCHK(DMGetCoordinateSection(dm, &coord_section));
  PETSCCHK(PetscSectionSetNumFields(coord_section, 1));
  PETSCCHK(PetscSectionSetFieldComponents(coord_section, 0, ndim));

  PETSCCHK(PetscSectionSetChart(coord_section, vertex_start, vertex_end));
  for (PetscInt v = vertex_start; v < vertex_end; ++v) {
    PETSCCHK(PetscSectionSetDof(coord_section, v, ndim));
    PETSCCHK(PetscSectionSetFieldDof(coord_section, v, 0, ndim));
  }
  PETSCCHK(PetscSectionSetUp(coord_section));
  */

  PetscInterface::setup_coordinate_section(dm, vertex_start, vertex_end);

  Vec coordinates;
  PetscScalar *coords;
  PetscInterface::setup_local_coordinate_vector(dm, coordinates);

  /*
  PetscSection coord_section;
  PETSCCHK(DMGetCoordinateSection(dm, &coord_section));

  // create the actual coordinates vector
  PetscInt coord_size;

  PETSCCHK(PetscSectionGetStorageSize(coord_section, &coord_size));
  PETSCCHK(VecCreate(PETSC_COMM_SELF, &coordinates));
  PETSCCHK(PetscObjectSetName((PetscObject)coordinates, "coordinates"));
  PETSCCHK(VecSetSizes(coordinates, coord_size, PETSC_DETERMINE));
  PETSCCHK(VecSetBlockSize(coordinates, ndim));
  PETSCCHK(VecSetType(coordinates, VECSTANDARD));
  */

  PETSCCHK(VecGetArray(coordinates, &coords));

  coords[0] = 0.0;
  coords[1] = 0.0;

  coords[2] = 1.0;
  coords[3] = 0.0;

  coords[4] = 1.0;
  coords[5] = 1.0;

  coords[6] = 0.0;
  coords[7] = 1.0;

  PETSCCHK(VecRestoreArray(coordinates, &coords));
  PETSCCHK(DMSetCoordinatesLocal(dm, coordinates));
  PETSCCHK(VecDestroy(&coordinates));

  // end of mesh creation

  PetscInt ndim_test;
  PETSCCHK(DMGetCoordinateDim(dm, &ndim_test));
  ASSERT_EQ(ndim_test, 2);

  DMPolytopeType celltype;
  PETSCCHK(DMPlexGetCellType(dm, 0, &celltype));
  ASSERT_EQ(celltype, DM_POLYTOPE_QUADRILATERAL);

  PetscInt coord_size;
  PetscSection coord_section;
  PETSCCHK(DMGetCoordinateSection(dm, &coord_section));
  PETSCCHK(PetscSectionGetStorageSize(coord_section, &coord_size));
  ASSERT_EQ(coord_size, 8);

  Vec v;
  PETSCCHK(VecCreate(MPI_COMM_SELF, &v));
  PETSCCHK(VecSetSizes(v, 4, 4));
  PETSCCHK(VecSetBlockSize(v, 2));
  PETSCCHK(VecSetFromOptions(v));

  PetscScalar *v_ptr;
  PETSCCHK(VecGetArrayWrite(v, &v_ptr));
  v_ptr[0] = 0.59;
  v_ptr[1] = 0.59;
  v_ptr[2] = 1.99;
  v_ptr[3] = 2.99;
  PETSCCHK(VecRestoreArrayWrite(v, &v_ptr));

  PetscSF cell_sf = nullptr;
  // PETSCCHK(PetscSFCreate(MPI_COMM_SELF, &cell_sf));
  PETSCCHK(DMLocatePoints(dm, v, DM_POINTLOCATION_NONE, &cell_sf));

  const PetscSFNode *cells;
  PetscInt n_found;
  const PetscInt *found;

  PETSCCHK(PetscSFGetGraph(cell_sf, NULL, &n_found, &found, &cells));
  ASSERT_EQ(n_found, 1);
  ASSERT_EQ(cells[0].index, 0);
  ASSERT_TRUE(cells[1].index < 0);

  PETSCCHK(VecDestroy(&v));
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

namespace {

inline DM get_simple_triangle() {
  DM dm;

  /*
   * 6
   * |\
   * 3  2
   * | 0 \
   * 4--1--5
   */

  PETSCCHK(DMCreate(MPI_COMM_WORLD, &dm));
  PETSCCHK(DMSetType(dm, DMPLEX));
  PETSCCHK(DMSetDimension(dm, 2));
  PETSCCHK(DMPlexSetChart(dm, 0, 7));

  PETSCCHK(DMPlexSetConeSize(dm, 0, 3));
  PETSCCHK(DMPlexSetConeSize(dm, 1, 2));
  PETSCCHK(DMPlexSetConeSize(dm, 2, 2));
  PETSCCHK(DMPlexSetConeSize(dm, 3, 2));
  PETSCCHK(DMSetUp(dm));

  std::vector<PetscInt> pts;
  pts = {1, 2, 3};
  PETSCCHK(DMPlexSetCone(dm, 0, pts.data()));
  pts = {4, 5};
  PETSCCHK(DMPlexSetCone(dm, 1, pts.data()));
  pts = {5, 6};
  PETSCCHK(DMPlexSetCone(dm, 2, pts.data()));
  pts = {6, 4};
  PETSCCHK(DMPlexSetCone(dm, 3, pts.data()));
  PETSCCHK(DMPlexSymmetrize(dm));
  PETSCCHK(DMPlexStratify(dm));

  // create coordinates section for dm
  const PetscInt ndim = 2;
  PETSCCHK(DMSetCoordinateDim(dm, ndim));
  const PetscInt vertex_start = 4;
  const PetscInt vertex_end = vertex_start + 3;
  PetscInterface::setup_coordinate_section(dm, vertex_start, vertex_end);

  Vec coordinates;
  PetscScalar *coords;
  PetscInterface::setup_local_coordinate_vector(dm, coordinates);
  PETSCCHK(VecGetArray(coordinates, &coords));

  coords[0] = 0.0;
  coords[1] = 0.0;

  coords[2] = 1.0;
  coords[3] = 0.0;

  coords[4] = 0.0;
  coords[5] = 1.0;

  PETSCCHK(VecRestoreArray(coordinates, &coords));
  PETSCCHK(DMSetCoordinatesLocal(dm, coordinates));
  PETSCCHK(VecDestroy(&coordinates));
  return dm;
}

inline DM get_simple_square() {
  DM dm;

  /*
   * 8--3--7
   * |     |
   * 4  0  2
   * |     |
   * 5--1--6
   */

  PETSCCHK(DMCreate(MPI_COMM_WORLD, &dm));
  PETSCCHK(DMSetType(dm, DMPLEX));
  PETSCCHK(DMSetDimension(dm, 2));
  PETSCCHK(DMPlexSetChart(dm, 0, 9));

  PETSCCHK(DMPlexSetConeSize(dm, 0, 4));
  PETSCCHK(DMPlexSetConeSize(dm, 1, 2));
  PETSCCHK(DMPlexSetConeSize(dm, 2, 2));
  PETSCCHK(DMPlexSetConeSize(dm, 3, 2));
  PETSCCHK(DMPlexSetConeSize(dm, 4, 2));
  PETSCCHK(DMSetUp(dm));

  std::vector<PetscInt> pts;
  pts = {1, 2, 3, 4};
  PETSCCHK(DMPlexSetCone(dm, 0, pts.data()));
  pts = {5, 6};
  PETSCCHK(DMPlexSetCone(dm, 1, pts.data()));
  pts = {6, 7};
  PETSCCHK(DMPlexSetCone(dm, 2, pts.data()));
  pts = {7, 8};
  PETSCCHK(DMPlexSetCone(dm, 3, pts.data()));
  pts = {8, 5};
  PETSCCHK(DMPlexSetCone(dm, 4, pts.data()));
  PETSCCHK(DMPlexSymmetrize(dm));
  PETSCCHK(DMPlexStratify(dm));

  // create coordinates section for dm
  const PetscInt ndim = 2;
  PETSCCHK(DMSetCoordinateDim(dm, ndim));
  const PetscInt vertex_start = 5;
  const PetscInt vertex_end = vertex_start + 4;

  PetscInterface::setup_coordinate_section(dm, vertex_start, vertex_end);

  Vec coordinates;
  PetscScalar *coords;
  PetscInterface::setup_local_coordinate_vector(dm, coordinates);
  PETSCCHK(VecGetArray(coordinates, &coords));

  coords[0] = 0.0;
  coords[1] = 0.0;

  coords[2] = 1.0;
  coords[3] = 0.0;

  coords[4] = 1.0;
  coords[5] = 1.0;

  coords[6] = 0.0;
  coords[7] = 1.0;

  PETSCCHK(VecRestoreArray(coordinates, &coords));
  PETSCCHK(DMSetCoordinatesLocal(dm, coordinates));
  PETSCCHK(VecDestroy(&coordinates));
  return dm;
}

} // namespace

TEST(PETSC, dm_triangle_mapping) {
  PETSCCHK(PetscInitializeNoArguments());
  auto dm = get_simple_triangle();
  PetscInterface::DMPlexHelper dmh(MPI_COMM_WORLD, dm);

  std::vector<PetscScalar> inner = {0.1, 0.1};
  ASSERT_TRUE(dmh.contains_point(inner) > -1);
  std::vector<PetscScalar> outer = {0.501, 0.501};
  ASSERT_TRUE(dmh.contains_point(outer) < 0);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSC, dm_cell_linearise) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm = get_simple_square();

  std::function<PetscInt(PetscInt)> rename_function = [&](PetscInt x) {
    return x;
  };

  auto spec = PetscInterface::get_cell_specification(dm, 0, rename_function);

  std::vector<std::byte> spec_buffer;
  spec.serialise(spec_buffer);
  PetscInterface::CellSTDRepresentation spec_copy;
  spec_copy.deserialise(spec_buffer);

  ASSERT_EQ(spec.vertices, spec_copy.vertices);
  ASSERT_EQ(spec.point_specs, spec_copy.point_specs);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST_P(PETSC_NDIM, dm_local_mapping) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  PetscInt ndim = GetParam();
  const int mesh_size = (ndim == 2) ? 31 : 17;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               /* lower */ NULL,
                               /* upper */ NULL,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const auto cell_count = mesh->get_cell_count();
  const int N = cell_count;

  ParticleSet initial_distribution(N, particle_spec);

  std::vector<double> point_in_domain(ndim);
  std::vector<REAL> point_tmp(ndim);
  mesh->get_point_in_subdomain(point_in_domain.data());
  for (int px = 0; px < N; px++) {
    // Get a point in the middle of the px-th cell.
    mesh->dmh->get_cell_vertex_average(px, point_tmp);

    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = point_tmp.at(dimx);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cell_count - 1;
  }

  A->add_particles_local(initial_distribution);

  mapper->map(*A);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto CELL_ID = A->get_cell(Sym<INT>("CELL_ID"), cellx);
    auto RANK = A->get_cell(Sym<INT>("NESO_MPI_RANK"), cellx);
    auto nrow = CELL_ID->nrow;
    for (int rx = 0; rx < nrow; rx++) {
      ASSERT_EQ(CELL_ID->at(rx, 0), rx);
      ASSERT_EQ(RANK->at(rx, 1), sycl_target->comm_pair.rank_parent);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST_P(PETSC_NDIM, dm_cart_advection) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  PetscInt ndim = GetParam();
  const int mesh_size = (ndim == 2) ? 7 : 17;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               /* lower */ NULL,
                               /* upper */ NULL,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const auto cell_count = mesh->get_cell_count();
  const int N = cell_count;

  ParticleSet initial_distribution(N, particle_spec);

  std::vector<double> point_in_domain(ndim);
  std::vector<REAL> point_tmp(ndim);
  mesh->get_point_in_subdomain(point_in_domain.data());

  REAL s = 1.0;
  const REAL h = 1.0 / mesh_size;

  for (int px = 0; px < N; px++) {
    // Get a point in the middle of the px-th cell.
    mesh->dmh->get_cell_vertex_average(px, point_tmp);

    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = point_tmp.at(dimx);
      initial_distribution[Sym<REAL>("V")][px][dimx] = s * h;
      s *= -1.0;
    }

    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cell_count - 1;
  }

  A->add_particles_local(initial_distribution);

  auto loop_advect = particle_loop(
      A,
      [=](auto P, auto V) {
        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) += V.at(dx);
        }
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")));

  auto loop_pbc = particle_loop(
      A,
      [=](auto P) {
        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) = fmod(P.at(dx) + 100.0, 1.0);
        }
      },
      Access::write(Sym<REAL>("P")));

  const int N_step = 10;
  for (int stepx = 0; stepx < N_step; stepx++) {
    loop_advect->execute();
    loop_pbc->execute();
    A->hybrid_move();
    A->cell_move();
  }

  A->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

class PETSC_HALO_OVERLAP : public testing::TestWithParam<int> {};
TEST_P(PETSC_HALO_OVERLAP, dm_halos) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const PetscInt ndim = 2;
  const int mesh_size = (ndim == 2) ? 31 : 17;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               /* lower */ NULL,
                               /* upper */ NULL,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInt overlap = GetParam();
  PetscInterface::generic_distribute(&dm, MPI_COMM_WORLD, overlap);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  ASSERT_TRUE(mesh->validate_halos(false));

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
INSTANTIATE_TEST_SUITE_P(init, PETSC_HALO_OVERLAP, testing::Values(0, 1));

TEST(PETSC, dmplex_project_evaluate) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/reference_all_types_square_0.2.msh");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  const int ndim = mesh->get_ndim();
  const int cell_count = mesh->get_cell_count();
  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  auto qpm = std::make_shared<ExternalCommon::QuadraturePointMapper>(
      sycl_target, domain);
  std::vector<std::vector<double>> positions;
  std::vector<int> cells;

  PetscInterface::uniform_within_dmplex_cells(mesh, 2, positions, cells);

  qpm->add_points_initialise();
  const int npart_qpm = positions.at(0).size();
  REAL point[2];
  for (int px = 0; px < npart_qpm; px++) {
    point[0] = positions.at(0).at(px);
    point[1] = positions.at(1).at(px);
    qpm->add_point(point);
  }
  qpm->add_points_finalise();

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<REAL>("R"), 1),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const int npart_per_cell = 17;
  PetscInterface::uniform_within_dmplex_cells(mesh, npart_per_cell, positions,
                                              cells);
  const int N = cells.size();
  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<REAL>("Q")][px][0] = cells.at(px) + 1.0;
  }
  A->add_particles_local(initial_distribution);

  auto dpe =
      std::make_shared<PetscInterface::DMPlexProjectEvaluate>(qpm, "DG", 0);
  dpe->project(A, Sym<REAL>("Q"));

  for (int cx = 0; cx < cell_count; cx++) {
    auto Q = qpm->particle_group->get_cell(qpm->get_sym(1), cx);
    for (int rx = 0; rx < Q->nrow; rx++) {
      ASSERT_NEAR(Q->at(rx, 0),
                  (cx + 1) * npart_per_cell / mesh->dmh->get_cell_volume(cx),
                  1.0e-10);
    }
  }

  // scale the quadrature point values
  const REAL k_scale = 3.14;
  particle_loop(
      qpm->particle_group, [=](auto Q) { Q.at(0) *= k_scale; },
      Access::write(Sym<REAL>(qpm->get_sym(1))))
      ->execute();

  // evalutate the scale quadrature point values
  dpe->evaluate(A, Sym<REAL>("R"));

  for (int cx = 0; cx < cell_count; cx++) {
    auto R = A->get_cell(Sym<REAL>("R"), cx);
    for (int rx = 0; rx < R->nrow; rx++) {
      const REAL correct =
          k_scale * (cx + 1) * npart_per_cell / mesh->dmh->get_cell_volume(cx);
      const REAL rescale = 1.0 / std::max(1.0, std::abs(correct));
      ASSERT_NEAR(R->at(rx, 0) * rescale, correct * rescale, 1.0e-10);
    }
  }

  A->free();
  qpm->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSC, dmplex_volume) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/reference_all_types_square_0.2.msh");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  const double volume = mesh->dmh->get_volume();
  ASSERT_NEAR(volume, 4.0, 1.0e-10);

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

INSTANTIATE_TEST_SUITE_P(init, PETSC_NDIM, testing::Values(2));

// TEST(PETSC, dm_all_types_3d) {
//   // TODO REMOVE?
//   PETSCCHK(PetscInitializeNoArguments());
//   DM dm;
//
//   std::string filename = "/home/js0259/git-ukaea/NESO-workspace/"
// "reference_all_types_cube/mixed_ref_cube_0.7.msh";
//
//   PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD, filename.c_str(),
//                                     (PetscBool)1, &dm));
//
//   PETSCCHK(DMDestroy(&dm));
//   PETSCCHK(PetscFinalize());
// }

TEST(PETSC, foo_internal) {
  // TODO REMOVE?
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = (ndim == 2) ? 32 : 16;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               /* lower */ NULL,
                               /* upper */ NULL,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  auto num_labels = mesh->dmh->get_num_labels();
  nprint("num_labels:", num_labels);
  for (int lx = 0; lx < num_labels; lx++) {
    nprint(lx, mesh->dmh->get_label_name(lx));
  }

  // auto map_face = mesh->dmh->get_face_sets();
  // for (auto &item : map_face) {
  //   nprint(item.first);
  //   for (auto px : item.second) {
  //     nprint("\t", px);
  //   }
  // }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSC, foo_gmsh) {
  // TODO REMOVE?
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/reference_all_types_square_0.2.msh");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));

  PetscInterface::generic_distribute(&dm);
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  ASSERT_TRUE(mesh->validate_halos());

  auto num_labels = mesh->dmh->get_num_labels();
  nprint("num_labels:", num_labels);
  for (int lx = 0; lx < num_labels; lx++) {
    nprint(lx, mesh->dmh->get_label_name(lx));
  }

  // auto map_face = mesh->dmh->get_face_sets();
  // for (auto &item : map_face) {
  //   nprint(item.first);
  //   for (auto px : item.second) {
  //     nprint("\t", px);
  //   }
  // }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
