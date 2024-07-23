#ifndef _NESO_PARTICLES_PETSC_UTILITY_HPP_
#define _NESO_PARTICLES_PETSC_UTILITY_HPP_

#include "petsc_common.hpp"
#include <petscblaslapack.h>

namespace NESO::Particles::PetscInterface {

/**
 * Invert a square matrix using a PETSc provided LU decomposition. This
 * function uses row-major format.
 *
 * @param[in] N number of rows/columns.
 * @param[in] A Input matrix to invert.
 * @param[in, out] X Allocated space for output matrix.
 */
template <typename T> inline void invert_matrix(const int N, const T *A, T *X) {

  Mat pA, pB, pX;

  std::vector<PetscScalar> hA(N * N);
  std::vector<PetscScalar> hB(N * N);
  std::vector<PetscScalar> hX(N * N);
  for (int rx = 0; rx < N; rx++) {
    for (int cx = 0; cx < N; cx++) {
      // Note the transpose as PETSc is using column-major.
      hA.at(rx * N + cx) = A[cx * N + rx];
      const PetscScalar value = (rx == cx) ? 1.0 : 0.0;
      hB.at(rx * N + cx) = value;
      hX.at(rx * N + cx) = 0.0;
    }
  }

  PETSCCHK(MatCreateSeqDense(PETSC_COMM_SELF, N, N, hA.data(), &pA));
  PETSCCHK(MatCreateSeqDense(PETSC_COMM_SELF, N, N, hB.data(), &pB));
  PETSCCHK(MatCreateSeqDense(PETSC_COMM_SELF, N, N, hX.data(), &pX));

  MatView(pA, PETSC_VIEWER_STDOUT_SELF);

  // In-place LU factorise A.
  PETSCCHK(MatLUFactor(pA, NULL, NULL, NULL));
  // Solve for the columns of B into X.
  PETSCCHK(MatMatSolve(pA, pB, pX));

  for (int rx = 0; rx < N; rx++) {
    for (int cx = 0; cx < N; cx++) {
      PetscScalar value;
      PETSCCHK(MatGetValue(pX, rx, cx, &value));
      X[rx * N + cx] = value;
    }
  }

  PETSCCHK(MatDestroy(&pA));
  PETSCCHK(MatDestroy(&pB));
  PETSCCHK(MatDestroy(&pX));
}

} // namespace NESO::Particles::PetscInterface

#endif
