#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>
#include <string>

using namespace NESO::Particles;

class PETSC_NDIM : public testing::TestWithParam<int> {};
TEST_P(PETSC_NDIM, init) {
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PetscInt ndim = GetParam();

  const int mesh_size = (ndim == 2) ? 32 : 16;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
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

TEST(PETSc, create_dm) {

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

TEST(PETSc, dm_triangle_mapping) {
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

TEST(PETSc, dm_cell_linearise) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm = get_simple_square();

  std::function<PetscInt(PetscInt)> rename_function = [&](PetscInt x) {
    return x;
  };

  auto spec = PetscInterface::CellSTDRepresentation(dm, 0, rename_function);

  std::vector<std::byte> spec_buffer;
  spec.serialise(spec_buffer);
  PetscInterface::CellSTDRepresentation spec_copy;
  spec_copy.deserialise(spec_buffer);

  ASSERT_EQ(spec.vertices, spec_copy.vertices);
  ASSERT_EQ(spec.point_cones, spec_copy.point_cones);
  ASSERT_EQ(spec.point_cone_orientations, spec_copy.point_cone_orientations);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST_P(PETSC_NDIM, dm_local_mapping) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  PetscInt ndim = GetParam();
  const int mesh_size = (ndim == 2) ? 31 : 17;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
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
  const int mesh_size = 17;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
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

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
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

TEST(PETSc, dmplex_volume) {
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

TEST(PETSc, dmplex_helper) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const PetscInt ndim = 2;
  const int mesh_size = (ndim == 2) ? 31 : 17;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
      /* lower */ NULL,
      /* upper */ NULL,
      /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm, MPI_COMM_WORLD, 1);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  int cell_count = mesh->get_cell_count();
  ASSERT_EQ(cell_count, mesh->dmh->get_cell_count());

  int size = 0;
  int rank = 0;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (size > 1) {

    int tmp = -1;

    MPICHK(
        MPI_Allreduce(&cell_count, &tmp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    ASSERT_EQ(mesh->dmh->get_global_cell_count(), tmp);

    PetscInt depth = 0;
    PETSCCHK(DMPlexGetDepth(dm, &depth));
    ASSERT_EQ(depth, 2);

    PetscInt gsize = 0;
    PETSCCHK(DMPlexGetDepthStratumGlobalSize(dm, depth, &gsize));

    ASSERT_EQ(static_cast<int>(gsize), tmp);
  }
  {
    auto [offset, global_owners] =
        PetscInterface::get_map_from_global_cell_points_to_ranks(dm);

    for (int cellx = 0; cellx < cell_count; cellx++) {
      const auto point_index = mesh->dmh->get_dmplex_cell_index(cellx);
      const auto global_point_index =
          mesh->dmh->get_point_global_index(point_index);
      const int rank_test = global_owners.at(global_point_index - offset);
      ASSERT_EQ(rank_test, rank);
    }
  }

  {

    PetscInt point_start = 0;
    PetscInt point_end = 0;
    PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));
    for (PetscInt px = point_start; px < point_end; px++) {
      const PetscInt global_point = mesh->dmh->get_point_global_index(px);
      const PetscInt local_point =
          mesh->dmh->get_local_point_from_global_point(global_point);
      ASSERT_EQ(local_point, px);
    }
  }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSc, dmplex_mesh_coupler_dg0) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  int size = 0;
  int rank = 0;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  const PetscInt ndim = 2;
  const int mesh_size = (ndim == 2) ? 31 : 17;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
      /* lower */ NULL,
      /* upper */ NULL,
      /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm, MPI_COMM_WORLD, 1);

  PetscInterface::DMPlexHelper dmh(MPI_COMM_WORLD, dm);
  const int cell_count_B = dmh.get_cell_count();
  const int cell_count_A = rank + 1;

  int tmp = 0;
  int tmp_offset = 0;
  for (int rx = 0; rx < size; rx++) {
    if (rank == rx) {
      tmp_offset = tmp;
    }
    tmp += rx + 1;
  }
  const int global_cell_count_A = tmp;
  const int dof_offset_A = tmp_offset;

  const int global_cell_count_B = dmh.get_global_cell_count();
  tmp = 0;
  MPICHK(MPI_Exscan(&cell_count_B, &tmp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  const int dof_offset_B = tmp;

  {

    std::vector<std::vector<PetscInterface::DMPlexMeshCouplerDG0MapEntry>>
        coupling_map(cell_count_A);

    auto cmdg0 = std::make_shared<PetscInterface::DMPlexMeshCouplerDG0>(
        dm, coupling_map);
    ASSERT_EQ(cmdg0->cell_count_A, cell_count_A);

    int indegree = -1;
    int outdegree = -1;
    int weighted = -1;
    MPICHK(MPI_Dist_graph_neighbors_count(cmdg0->comm_forward, &indegree,
                                          &outdegree, &weighted));
    ASSERT_EQ(indegree, 0);
    ASSERT_EQ(outdegree, 0);

    MPICHK(MPI_Dist_graph_neighbors_count(cmdg0->comm_backward, &indegree,
                                          &outdegree, &weighted));
    ASSERT_EQ(indegree, 0);
    ASSERT_EQ(outdegree, 0);
  }

  {
    std::set<int> local_cell_indices;
    for (PetscInt cellx = 0; cellx < cell_count_B; cellx++) {
      const PetscInt cell_point_index = dmh.get_dmplex_cell_index(cellx);
      const int cell_point_global_index =
          dmh.get_point_global_index(cell_point_index);
      local_cell_indices.insert(cell_point_global_index);
    }

    std::set<int> global_cell_indices_set =
        set_all_reduce_union(local_cell_indices, MPI_COMM_WORLD);

    ASSERT_EQ(static_cast<int>(global_cell_indices_set.size()),
              dmh.get_global_cell_count());

    std::vector<int> global_cell_indices;
    global_cell_indices.reserve(global_cell_indices_set.size());
    for_each(global_cell_indices_set.begin(), global_cell_indices_set.end(),
             [&](auto ix) { global_cell_indices.push_back(ix); });

    // deliberatly the same seed on all ranks
    std::mt19937 rng(5234234);
    std::uniform_int_distribution<int> dist_cells(
        0, static_cast<int>(global_cell_indices.size() - 1));
    std::uniform_int_distribution<int> dist_number(0, 4);
    std::uniform_real_distribution<REAL> dist_weight(1.0, 2.0);

    std::map<
        int,
        std::vector<std::vector<PetscInterface::DMPlexMeshCouplerDG0MapEntry>>>
        map_rank_to_map;

    for (int rankx = 0; rankx < size; rankx++) {

      const int num_owned_cells = rankx + 1;
      map_rank_to_map[rankx].resize(num_owned_cells);

      for (int cellx = 0; cellx < num_owned_cells; cellx++) {
        const int map_size = dist_number(rng);
        std::set<int> tmp_cells_set;
        for (int mx = 0; mx < map_size; mx++) {
          int tmp_cell = 0;
          do {
            tmp_cell = dist_cells(rng);
          } while (tmp_cells_set.count(tmp_cell));
          tmp_cells_set.insert(tmp_cell);
          map_rank_to_map.at(rankx).at(cellx).push_back(
              {tmp_cell, dist_weight(rng), dist_weight(rng)});
        }
      }
    }

    // compute the outward edges and inward edges on this rank
    std::set<int> edges_forwards;
    std::set<int> edges_backwards;

    auto [offset, global_owners] =
        PetscInterface::get_map_from_global_cell_points_to_ranks(dm);

    auto &maps = map_rank_to_map.at(rank);
    for (int cellx = 0; cellx < cell_count_A; cellx++) {
      for (auto &ex : maps.at(cellx)) {
        const int cell_index = ex.cell_index;
        const int owning_rank = global_owners.at(cell_index - offset);
        edges_forwards.insert(owning_rank);
      }
    }

    for (int rankx = 0; rankx < size; rankx++) {

      for (auto cell_map : map_rank_to_map.at(rankx)) {
        for (auto &ex : cell_map) {
          const int cell_index = ex.cell_index;
          const int owning_rank = global_owners.at(cell_index - offset);
          if (rank == owning_rank) {
            edges_backwards.insert(rankx);
          }
        }
      }
    }

    auto cmdg0 = std::make_shared<PetscInterface::DMPlexMeshCouplerDG0>(
        dm, map_rank_to_map.at(rank));

    int indegree = -1;
    int outdegree = -1;
    int weighted = -1;
    MPICHK(MPI_Dist_graph_neighbors_count(cmdg0->comm_forward, &indegree,
                                          &outdegree, &weighted));
    ASSERT_EQ(outdegree, static_cast<int>(edges_forwards.size()));
    ASSERT_EQ(indegree, static_cast<int>(edges_backwards.size()));

    std::vector<int> sources(indegree);
    std::vector<int> destinations(outdegree);
    std::vector<int> source_weights(indegree);
    std::vector<int> destination_weights(outdegree);

    SuppressMPINullPtrCheck snpc;
    MPICHK(MPI_Dist_graph_neighbors(cmdg0->comm_forward, indegree,
                                    snpc.get(sources), snpc.get(source_weights),
                                    outdegree, snpc.get(destinations),
                                    snpc.get(destination_weights)));

    std::set<int> tmp_set;
    for (int rx = 0; rx < indegree; rx++) {
      const int rankt = sources.at(rx);
      ASSERT_EQ(edges_backwards.count(rankt), 1);
      tmp_set.insert(rankt);
    }
    ASSERT_EQ(tmp_set, edges_backwards);

    tmp_set.clear();
    for (int rx = 0; rx < outdegree; rx++) {
      const int rankt = destinations.at(rx);
      ASSERT_EQ(edges_forwards.count(rankt), 1);
      tmp_set.insert(rankt);
    }
    ASSERT_EQ(tmp_set, edges_forwards);

    MPICHK(MPI_Dist_graph_neighbors_count(cmdg0->comm_backward, &indegree,
                                          &outdegree, &weighted));
    ASSERT_EQ(indegree, static_cast<int>(edges_forwards.size()));
    ASSERT_EQ(outdegree, static_cast<int>(edges_backwards.size()));

    sources.resize(indegree);
    destinations.resize(outdegree);
    source_weights.resize(indegree);
    destination_weights.resize(outdegree);

    MPICHK(MPI_Dist_graph_neighbors(cmdg0->comm_backward, indegree,
                                    snpc.get(sources), snpc.get(source_weights),
                                    outdegree, snpc.get(destinations),
                                    snpc.get(destination_weights)));
    tmp_set.clear();
    for (int rx = 0; rx < indegree; rx++) {
      const int rankt = sources.at(rx);
      ASSERT_EQ(edges_forwards.count(rankt), 1);
      tmp_set.insert(rankt);
    }
    ASSERT_EQ(tmp_set, edges_forwards);

    tmp_set.clear();
    for (int rx = 0; rx < outdegree; rx++) {
      const int rankt = destinations.at(rx);
      ASSERT_EQ(edges_backwards.count(rankt), 1);
      tmp_set.insert(rankt);
    }
    ASSERT_EQ(tmp_set, edges_backwards);

    tmp_set.clear();
    std::for_each(cmdg0->sources_forward.begin(), cmdg0->sources_forward.end(),
                  [&](auto ix) { tmp_set.insert(ix); });
    ASSERT_EQ(tmp_set, edges_backwards);
    tmp_set.clear();
    std::for_each(cmdg0->destinations_forward.begin(),
                  cmdg0->destinations_forward.end(),
                  [&](auto ix) { tmp_set.insert(ix); });
    ASSERT_EQ(tmp_set, edges_forwards);

    tmp_set.clear();
    std::for_each(cmdg0->sources_backward.begin(),
                  cmdg0->sources_backward.end(),
                  [&](auto ix) { tmp_set.insert(ix); });
    ASSERT_EQ(tmp_set, edges_forwards);
    tmp_set.clear();
    std::for_each(cmdg0->destinations_backward.begin(),
                  cmdg0->destinations_backward.end(),
                  [&](auto ix) { tmp_set.insert(ix); });
    ASSERT_EQ(tmp_set, edges_backwards);

    PetscInterface::DMPlexHelper dmh_B{MPI_COMM_WORLD, dm};
    auto lambda_global_to_local = [&](std::vector<int> &cells) {
      std::for_each(cells.begin(), cells.end(), [&](auto &cellx) {
        cellx = dmh_B.get_local_point_from_global_point(cellx);
      });
    };

    // ranks that will send to this rank
    int index = 0;
    int test_total_number_cells = 0;
    for (const int rankx : cmdg0->sources_forward) {
      // the local copy of the map for that rank
      auto &m = map_rank_to_map.at(rankx);

      std::vector<int> cells_correct;

      // loop over cells on the rank
      for (auto &entries_outer : m) {
        for (auto &ex : entries_outer) {
          const int cell_index = ex.cell_index;
          const int owning_rank = global_owners.at(cell_index - offset);
          if (rank == owning_rank) {
            cells_correct.push_back(cell_index);
          }
        }
      }

      const int soffset = static_cast<int>(cmdg0->recv_disps_forward.at(index));
      const int num_cells = static_cast<int>(cells_correct.size());
      test_total_number_cells += num_cells;

      lambda_global_to_local(cells_correct);

      for (int ix = 0; ix < num_cells; ix++) {
        ASSERT_EQ(cells_correct.at(ix),
                  cmdg0->cells_forward_B.at(soffset + ix));
      }
      index++;
    }
    ASSERT_EQ(test_total_number_cells, cmdg0->total_num_recv_cells_forward);

    int soffset = 0;
    test_total_number_cells = 0;
    for (const int rankx : cmdg0->destinations_forward) {
      auto &m = map_rank_to_map.at(rank);

      std::vector<int> cells_correct;
      std::vector<REAL> weights_correct;

      int index_A = 0;
      for (auto &entries_outer : m) {
        for (auto &ex : entries_outer) {
          const int cell_index = ex.cell_index;
          const int owning_rank = global_owners.at(cell_index - offset);
          if (rankx == owning_rank) {
            cells_correct.push_back(index_A);
            weights_correct.push_back(ex.weight_forward);
          }
        }
        index_A++;
      }

      const int num_cells = static_cast<int>(cells_correct.size());
      test_total_number_cells += num_cells;
      for (int ix = 0; ix < num_cells; ix++) {
        ASSERT_EQ(cells_correct.at(ix),
                  cmdg0->cells_forward_A.at(soffset + ix));
        ASSERT_EQ(weights_correct.at(ix),
                  cmdg0->weights_forward_A.at(soffset + ix));
      }
      soffset += num_cells;
    }
    ASSERT_EQ(test_total_number_cells, cmdg0->total_num_send_cells_forward);

    // backward sources are the ranks that hold B cells for our map
    auto &m = map_rank_to_map.at(rank);
    int rank_index = 0;
    test_total_number_cells = 0;
    for (const int rankx : cmdg0->sources_backward) {
      std::vector<int> cells_correct;
      std::vector<REAL> weights_correct;

      int index = 0;
      for (auto &cell : m) {
        for (auto &ex : cell) {
          const int cell_index = ex.cell_index;
          const int owning_rank = global_owners.at(cell_index - offset);
          if (rankx == owning_rank) {
            cells_correct.push_back(index);
            weights_correct.push_back(ex.weight_backward);
          }
        }
        index++;
      }

      const int num_cells = static_cast<int>(cells_correct.size());
      test_total_number_cells += num_cells;
      const int offsets =
          static_cast<int>(cmdg0->recv_disps_backward.at(rank_index));

      for (int ix = 0; ix < num_cells; ix++) {
        ASSERT_EQ(cmdg0->cells_backward_A.at(offsets + ix),
                  cells_correct.at(ix));
        ASSERT_EQ(cmdg0->weights_backward_A.at(offsets + ix),
                  weights_correct.at(ix));
      }

      rank_index++;
    }
    ASSERT_EQ(test_total_number_cells, cmdg0->total_num_recv_cells_backward);

    rank_index = 0;
    test_total_number_cells = 0;
    for (const int rankx : cmdg0->destinations_backward) {
      auto &m = map_rank_to_map.at(rankx);
      std::vector<int> cells_correct;
      for (auto &cell : m) {
        for (auto &ex : cell) {
          const int cell_index = ex.cell_index;
          const int owning_rank = global_owners.at(cell_index - offset);
          if (rank == owning_rank) {
            cells_correct.push_back(cell_index);
          }
        }
      }
      const int num_cells = static_cast<int>(cells_correct.size());
      test_total_number_cells += num_cells;
      const int offsets =
          static_cast<int>(cmdg0->send_disps_backward.at(rank_index));

      lambda_global_to_local(cells_correct);

      for (int ix = 0; ix < num_cells; ix++) {
        ASSERT_EQ(cmdg0->cells_backward_B.at(offsets + ix),
                  cells_correct.at(ix));
      }
      rank_index++;
    }
    ASSERT_EQ(test_total_number_cells, cmdg0->total_num_send_cells_backward);

    // Can now actually test the forward/backward transport
    std::vector<REAL> dofs_A_correct(cell_count_A);
    std::vector<REAL> dofs_A_to_test(cell_count_A);
    std::vector<REAL> dofs_B_correct(cell_count_B);
    std::vector<REAL> dofs_B_to_test(cell_count_B);
    std::vector<REAL> global_dofs_A(global_cell_count_A);
    std::vector<REAL> global_dofs_B(global_cell_count_B);

    auto lambda_reset_dofs = [&]() {
      std::fill(dofs_A_correct.begin(), dofs_A_correct.end(), 0.0);
      std::fill(dofs_A_to_test.begin(), dofs_A_to_test.end(), 0.0);
      std::fill(dofs_B_correct.begin(), dofs_B_correct.end(), 0.0);
      std::fill(dofs_B_to_test.begin(), dofs_B_to_test.end(), 0.0);
    };

    std::uniform_real_distribution<REAL> dist_dofs(1.0, 2.0);

    auto lambda_generate_dofs = [&](auto &dofs) {
      std::for_each(dofs.begin(), dofs.end(),
                    [&](auto &dx) { dx = dist_dofs(rng); });
    };

    lambda_reset_dofs();
    lambda_generate_dofs(global_dofs_A);
    std::copy(global_dofs_A.begin() + dof_offset_A,
              global_dofs_A.begin() + dof_offset_A + cell_count_A,
              dofs_A_correct.begin());

    const int point_index_offset = offset;
    auto lambda_forward_transfer = [&](auto &dofs_A, auto &dofs_B) {
      std::fill(dofs_B.begin(), dofs_B.end(), 0.0);
      std::vector<REAL> tmp_dofs_B(global_cell_count_B);
      std::fill(tmp_dofs_B.begin(), tmp_dofs_B.end(), 0.0);
      std::vector<REAL> tmp_dofs_B2(global_cell_count_B);
      std::fill(tmp_dofs_B2.begin(), tmp_dofs_B2.end(), 0.0);

      auto &m = map_rank_to_map.at(rank);

      int index = 0;
      for (auto &cell_entries : m) {
        const REAL contrib = dofs_A.at(index);
        for (auto &entry : cell_entries) {
          const int B_point_index = entry.cell_index;
          const REAL B_weight = entry.weight_forward;
          const int B_dof_index = B_point_index - point_index_offset;
          tmp_dofs_B.at(B_dof_index) += B_weight * contrib;
        }
        index++;
      }

      ASSERT_TRUE(static_cast<int>(dofs_B.size()) >= cell_count_B);
      MPICHK(MPI_Allreduce(tmp_dofs_B.data(), tmp_dofs_B2.data(),
                           global_cell_count_B, map_ctype_mpi_type<REAL>(),
                           MPI_SUM, MPI_COMM_WORLD));

      std::copy(tmp_dofs_B2.begin() + dof_offset_B,
                tmp_dofs_B2.begin() + dof_offset_B + cell_count_B,
                dofs_B.begin());
    };

    lambda_forward_transfer(dofs_A_correct, dofs_B_correct);
    cmdg0->forward_transfer(dofs_A_correct, dofs_B_to_test);

    for (int ix = 0; ix < cell_count_B; ix++) {
      ASSERT_NEAR(dofs_B_correct.at(ix), dofs_B_to_test.at(ix), 1.0e-12);
    }

    lambda_reset_dofs();
    lambda_generate_dofs(global_dofs_B);

    auto lambda_backward_transfer = [&](auto &global_dofs_B, auto &dofs_A) {
      std::fill(dofs_A.begin(), dofs_A.end(), 0.0);
      auto &m = map_rank_to_map.at(rank);

      int index = 0;
      for (auto &cell_entries : m) {
        for (auto &entry : cell_entries) {
          const int B_point_index = entry.cell_index;
          const REAL B_weight = entry.weight_backward;
          const int B_dof_index = B_point_index - point_index_offset;
          const REAL contrib = global_dofs_B.at(B_dof_index);
          dofs_A.at(index) += B_weight * contrib;
        }
        index++;
      }
    };

    std::copy(global_dofs_B.begin() + dof_offset_B,
              global_dofs_B.begin() + dof_offset_B + cell_count_B,
              dofs_B_correct.begin());

    lambda_backward_transfer(global_dofs_B, dofs_A_correct);
    cmdg0->backward_transfer(dofs_B_correct, dofs_A_to_test);

    for (int ix = 0; ix < cell_count_A; ix++) {
      ASSERT_NEAR(dofs_A_correct.at(ix), dofs_A_to_test.at(ix), 1.0e-12);
    }
  }

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSc, dmplex_mesh_coupler_dg0_integration) {

  PETSCCHK(PetscInitializeNoArguments());

  DM dm_outer;
  DM dm_inner;

  int size = 0;
  int rank = 0;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  MPI_Comm comm_outer = MPI_COMM_WORLD;
  MPI_Comm comm_inner = MPI_COMM_WORLD;

  const int inner_comm_size = 8;
  const int colour = (rank < inner_comm_size) ? 1 : MPI_UNDEFINED;
  const bool inner_active = colour != MPI_UNDEFINED;
  MPICHK(MPI_Comm_split(comm_outer, colour, rank, &comm_inner));

  const REAL cell_width_outer = 1.0;
  const REAL cell_width_inner = 2.0 * cell_width_outer;
  const PetscInt ndim = 2;
  const PetscInt mesh_size_inner = 4;
  const PetscInt mesh_offset = 2;
  const PetscInt mesh_size_outer = 2 * mesh_size_inner + 4 * mesh_offset;

  PetscInt faces_outer[2] = {mesh_size_outer, mesh_size_outer};
  PetscReal upper_outer[2] = {mesh_size_outer * cell_width_outer,
                              mesh_size_outer * cell_width_outer};
  PetscReal lower_outer[2] = {0.0, 0.0};

  PetscInt faces_inner[2] = {mesh_size_inner, mesh_size_inner};
  PetscReal upper_inner[2] = {
      mesh_size_inner * cell_width_inner + mesh_offset * cell_width_inner,
      mesh_size_inner * cell_width_inner + mesh_offset * cell_width_inner};
  PetscReal lower_inner[2] = {mesh_offset * cell_width_inner,
                              mesh_offset * cell_width_inner};

  PetscSF sf_outer;
  PetscSF sf_inner;

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      comm_outer, ndim, PETSC_FALSE, faces_outer, lower_outer, upper_outer,
      NULL, PETSC_TRUE, &dm_outer));
  PetscInterface::generic_distribute(&dm_outer, comm_outer, 1, &sf_outer);

  // PetscViewer viewer;
  // PETSCCHK(PetscViewerCreate(PETSC_COMM_SELF, &viewer));
  // PETSCCHK(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  // PETSCCHK(PetscSFView(sf_outer, viewer));

  PetscInt nroots = 0;
  PetscInt nleaves = 0;
  PetscInt const *ilocal = nullptr;
  PetscSFNode const *iremote = nullptr;
  PETSCCHK(PetscSFGetGraph(sf_outer, &nroots, &nleaves, &ilocal, &iremote));

  for (int ix = 0; ix < nleaves; ix++) {
    nprint(ix, iremote[ix].rank, iremote[ix].index);
  }

  nprint_variable(nroots);
  nprint_variable(nleaves);
  nprint_variable(ilocal);
  nprint_variable(iremote);

  if (inner_active) {
    PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
        comm_inner, ndim, PETSC_FALSE, faces_inner, lower_inner, upper_inner,
        NULL, PETSC_TRUE, &dm_inner));
    PetscInterface::generic_distribute(&dm_inner, comm_inner, 1, &sf_inner);
  }

  std::shared_ptr<PetscInterface::DMPlexHelper> helper_outer;
  std::shared_ptr<PetscInterface::DMPlexHelper> helper_inner;

  helper_outer =
      std::make_shared<PetscInterface::DMPlexHelper>(comm_outer, dm_outer);
  helper_outer->write_vtk("outer.vtu");

  if (inner_active) {
    helper_inner =
        std::make_shared<PetscInterface::DMPlexHelper>(comm_inner, dm_inner);
    helper_inner->write_vtk("inner.vtu");
  }

  const int cell_count_outer = helper_outer->get_cell_count();
  const int cell_count_inner =
      inner_active ? helper_inner->get_cell_count() : 0;

  std::vector<REAL> dofs_outer_forward(cell_count_outer);
  std::vector<REAL> dofs_inner_forward(cell_count_inner);
  std::vector<REAL> dofs_outer_backward(cell_count_outer);
  std::vector<REAL> dofs_inner_backward(cell_count_inner);

  for (int cellx = 0; cellx < cell_count_outer; cellx++) {
    const PetscInt petsc_cell_index =
        helper_outer->get_dmplex_cell_index(cellx);
    const PetscInt global_cell_index =
        helper_outer->get_point_global_index(petsc_cell_index);
    dofs_outer_backward.at(cellx) = global_cell_index;
    dofs_outer_forward.at(cellx) = 0.0;
  }

  for (int cellx = 0; cellx < cell_count_inner; cellx++) {
    const PetscInt petsc_cell_index =
        helper_inner->get_dmplex_cell_index(cellx);
    const PetscInt global_cell_index =
        helper_inner->get_point_global_index(petsc_cell_index);
    dofs_inner_backward.at(cellx) = 0.0;
    dofs_inner_forward.at(cellx) = global_cell_index;
  }

  std::vector<std::vector<PetscInterface::DMPlexMeshCouplerDG0MapEntry>>
      coupler_map(cell_count_inner);

  for (int cellx = 0; cellx < cell_count_inner; cellx++) {
    const PetscInt petsc_cell_index =
        helper_inner->get_dmplex_cell_index(cellx);
    const PetscInt global_cell_index =
        helper_inner->get_point_global_index(petsc_cell_index);
    nprint_variable(cellx);
    nprint_variable(global_cell_index);

    const PetscInt cell_index_x = global_cell_index % mesh_size_inner;
    const PetscInt cell_index_y = global_cell_index / mesh_size_inner;

    const PetscInt cell_index_outer_base_x = (mesh_offset + cell_index_x) * 2;
    const PetscInt cell_index_outer_base_y = (mesh_offset + cell_index_y) * 2;

    for (int dy : {0, 1}) {
      for (int dx : {0, 1}) {
        const PetscInt cell_index_outer_x = cell_index_outer_base_x + dx;
        const PetscInt cell_index_outer_y = cell_index_outer_base_y + dy;
        const PetscInt cell_index_outer =
            cell_index_outer_y * mesh_size_outer + cell_index_outer_x;
        coupler_map.at(cellx).push_back({cell_index_outer, 1.0, 0.25});
      }
    }
  }

  auto coupler = std::make_shared<PetscInterface::DMPlexMeshCouplerDG0>(
      dm_outer, coupler_map);

  coupler->forward_transfer(dofs_inner_forward, dofs_outer_forward);
  coupler->backward_transfer(dofs_outer_backward, dofs_inner_backward);

  auto vtk_outer = helper_outer->get_vtk_cell_data();
  for (int cellx = 0; cellx < cell_count_outer; cellx++) {
    vtk_outer.at(cellx).cell_data["forward"] = dofs_outer_forward.at(cellx);
    vtk_outer.at(cellx).cell_data["backward"] = dofs_outer_backward.at(cellx);
    vtk_outer.at(cellx).cell_data["rank"] = rank;

    const PetscInt petsc_cell_index =
        helper_outer->get_dmplex_cell_index(cellx);
    const PetscInt global_cell_index =
        helper_outer->get_point_global_index(petsc_cell_index);
    vtk_outer.at(cellx).cell_data["cell_index"] = global_cell_index;
  }
  VTK::VTKHDF vtkhdf_outer("outer.vtkhdf", comm_outer);
  vtkhdf_outer.write(vtk_outer);
  vtkhdf_outer.close();

  if (inner_active) {
    auto vtk_inner = helper_inner->get_vtk_cell_data();
    for (int cellx = 0; cellx < cell_count_inner; cellx++) {
      vtk_inner.at(cellx).cell_data["forward"] = dofs_inner_forward.at(cellx);
      vtk_inner.at(cellx).cell_data["backward"] = dofs_inner_backward.at(cellx);
      vtk_inner.at(cellx).cell_data["rank"] = rank;

      const PetscInt petsc_cell_index =
          helper_inner->get_dmplex_cell_index(cellx);
      const PetscInt global_cell_index =
          helper_inner->get_point_global_index(petsc_cell_index);
      vtk_inner.at(cellx).cell_data["cell_index"] = global_cell_index;
    }
    VTK::VTKHDF vtkhdf_inner("inner.vtkhdf", comm_inner);
    vtkhdf_inner.write(vtk_inner);
    vtkhdf_inner.close();
  }

  PETSCCHK(PetscSFDestroy(&sf_outer));
  PETSCCHK(DMDestroy(&dm_outer));
  if (inner_active) {
    PETSCCHK(DMDestroy(&dm_inner));
    PETSCCHK(PetscSFDestroy(&sf_inner));
  }

  if (comm_inner != MPI_COMM_NULL) {
    MPICHK(MPI_Comm_free(&comm_inner));
    comm_inner = MPI_COMM_NULL;
  }

  PETSCCHK(PetscFinalize());
}

#endif
