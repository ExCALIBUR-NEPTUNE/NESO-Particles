#include "gtest/gtest.h"
#include <iostream>
#include <mpi.h>

/*
 *  If an exception is thrown try and abort MPI cleanly to prevent a deadlock.
 */
void TerminateHandler() {
  std::cout << "Exception thrown attempting to abort cleanly." << std::endl;
  MPI_Abort(MPI_COMM_WORLD, -2);
}

int main(int argc, char **argv) {

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }

#if GTEST_HAS_EXCEPTIONS
  std::set_terminate(&TerminateHandler);
#endif
  ::testing::InitGoogleTest(&argc, argv);
  int err = RUN_ALL_TESTS();

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return err;
}
