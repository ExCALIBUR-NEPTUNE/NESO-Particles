#include <CL/sycl.hpp>
#include <catch2/catch.hpp>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

TEST_CASE("test_particle_dat_append_1") {

    SYCLTarget sycl_target{GPU_SELECTOR, MPI_COMM_WORLD};

    const int cell_count = 4;
    const int ncomp = 3;

    auto A = ParticleDat(
        sycl_target, ParticleProp(Sym<INT>("FOO"), ncomp), cell_count);

    std::vector<INT> counts(cell_count);
    for (int cellx = 0; cellx < cell_count; cellx++) {
        REQUIRE(A->s_npart_cell[cellx] == 0);
        counts[cellx] = 0;
    }

    const int N = 42;

    std::vector<INT> cells0(N);
    std::vector<INT> data0(N * ncomp);

    std::mt19937 rng(523905);
    std::uniform_int_distribution<int> cell_rng(0, cell_count - 1);

    INT index = 0;
    for (int px = 0; px < N; px++) {
        auto cell = cell_rng(rng);
        cells0[px] = cell;
        counts[cell]++;
        for (int cx = 0; cx < ncomp; cx++) {
            data0[cx * N + px] = ++index;
        }
    }
    A->realloc(counts);
    for (int cellx = 0; cellx < cell_count; cellx++) {
        REQUIRE(A->cell_dat.nrow[cellx] >= counts[cellx]);
    }

    A->append_particle_data(N, false, cells0, data0);
    // the append is async
    sycl_target.queue.wait();

    for (int cellx = 0; cellx < cell_count; cellx++) {
        REQUIRE(A->s_npart_cell[cellx] == counts[cellx]);
        // the "data exists" flag is false so these new values should all be
        // zero
        auto cell_data = A->cell_dat.get_cell(cellx);
        for (int cx = 0; cx < ncomp; cx++) {
            for (int px = 0; px < (A->s_npart_cell[cellx]); px++) {
                REQUIRE((*cell_data)[cx][px] == 0);
            }
        }
    }

    // append the same counts with actual data
    for (int cellx = 0; cellx < cell_count; cellx++) {
        counts[cellx] *= 2;
    }
    A->realloc(counts);

    for (int cellx = 0; cellx < cell_count; cellx++) {
        REQUIRE(A->cell_dat.nrow[cellx] >= counts[cellx]);
    }

    A->append_particle_data(N, true, cells0, data0);
    // the append is async
    sycl_target.queue.wait();

    for (int cellx = 0; cellx < cell_count; cellx++) {
        REQUIRE(A->s_npart_cell[cellx] == counts[cellx]);
        // the "data exists" flag is false so these new values should all be
        // zero
        auto cell_data = A->cell_dat.get_cell(cellx);

        // TODO need to verify the rows are copied into the correct cells
    }
}
