#include <external_interfaces/petsc/petsc_interface.hpp>
#ifdef NESO_PARTICLES_PETSC

#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <string>

using namespace NESO::Particles;

// TODO parameterise
TEST(PETSC, init) {
  PETSCCHK(PetscInitializeNoArguments());
  // TODO
  std::string gmsh_filename = "/home/js0259/git-ukaea/NESO-Particles-paper/"
                              "resources/mesh_ring/mesh_ring.msh";
  DM dm;
  // PETSCCHK(DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, gmsh_filename.c_str(),
  //                                   static_cast<PetscBool>(1), &dm));

  const int mesh_size = 32;
  PetscInt faces[2] = {mesh_size, mesh_size};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, faces,
                               /* lower */ NULL,
                               /* upper */ NULL,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  PetscInt start, end;
  PETSCCHK(DMPlexGetHeightStratum(dm, 0, &start, &end));
  ASSERT_EQ(mesh->get_cell_count(), end - start);
  ASSERT_EQ(mesh->get_ndim(), 2);

  auto mh = mesh->get_mesh_hierarchy();

  ASSERT_EQ(mh->dims[0], 1);
  ASSERT_EQ(mh->dims[1], 1);
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
    std::vector<REAL> cell_average(2);
    mesh->dmh->get_cell_vertex_average(cellx, cell_average);
    auto bb = mesh->dmh->get_cell_bounding_box(cellx);
    ASSERT_TRUE(bb->contains_point(2, cell_average));

    // Check cell claimed the corresponding MH cell
    INT mh_tuple[6];
    mapper.map_to_tuple(cell_average.data(), mh_tuple);
    const INT mh_linear_index = mapper.tuple_to_linear_global(mh_tuple);
    const int owner = mh->get_owner(mh_linear_index);
    ASSERT_EQ(owner, rank);
  }

  Vec v;
  PETSCCHK(VecCreate(MPI_COMM_SELF, &v));
  PETSCCHK(VecSetSizes(v, 4, 4));
  PETSCCHK(VecSetBlockSize(v, 2));
  PETSCCHK(VecSetFromOptions(v));

  PetscScalar *v_ptr;
  PETSCCHK(VecGetArrayWrite(v, &v_ptr));
  v_ptr[0] = 0.59;
  v_ptr[1] = 0.59;
  v_ptr[2] = 0.99;
  v_ptr[3] = 0.99;
  PETSCCHK(VecRestoreArrayWrite(v, &v_ptr));
  PETSCCHK(VecView(v, PETSC_VIEWER_STDOUT_SELF));

  PetscSF cell_sf = nullptr;
  // PETSCCHK(PetscSFCreate(MPI_COMM_SELF, &cell_sf));
  PETSCCHK(DMLocatePoints(dm, v, DM_POINTLOCATION_NONE, &cell_sf));

  PETSCCHK(PetscSFView(cell_sf, PETSC_VIEWER_STDOUT_SELF));
  const PetscSFNode *cells;
  PetscInt n_found;
  const PetscInt *found;

  PETSCCHK(PetscSFGetGraph(cell_sf, NULL, &n_found, &found, &cells));
  nprint("nfound:", n_found);
  nprint("found:", found);
  for (int nx = 0; nx < 2; nx++) {
    nprint("nx:", nx, "cell:", cells[nx].rank, cells[nx].index);
  }

  PETSCCHK(VecView(v, PETSC_VIEWER_STDOUT_SELF));

  // PETSCCHK(PetscSFDestroy(&cell_sf));
  PETSCCHK(VecDestroy(&v));

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
  PETSCCHK(VecView(v, PETSC_VIEWER_STDOUT_SELF));

  PetscSF cell_sf = nullptr;
  // PETSCCHK(PetscSFCreate(MPI_COMM_SELF, &cell_sf));
  PETSCCHK(DMLocatePoints(dm, v, DM_POINTLOCATION_NONE, &cell_sf));

  PETSCCHK(PetscSFView(cell_sf, PETSC_VIEWER_STDOUT_SELF));
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

TEST(PETSC, dm_halo_creation) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  // TODO
  std::string gmsh_filename = "/home/js0259/git-ukaea/NESO-Particles-paper/"
                              "resources/mesh_ring/mesh_ring.msh";
  PETSCCHK(DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, gmsh_filename.c_str(),
                                    PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  const int ndim = 2;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  const int N = 1;

  ParticleSet initial_distribution(N, particle_spec);

  std::vector<double> point_in_domain(ndim);
  mesh->get_point_in_subdomain(point_in_domain.data());
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = point_in_domain.at(dimx);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] =
        mesh->get_cell_count() - 1;
  }

  A->add_particles_local(initial_distribution);

  A->print(Sym<REAL>("P"), Sym<INT>("CELL_ID"));

  mapper->map(*A);
  A->print(Sym<INT>("CELL_ID"));

  A->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
