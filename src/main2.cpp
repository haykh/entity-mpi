#if defined(MPI_ENABLED)
#  include <mpi.h>
#endif

#include <Kokkos_Core.hpp>

#include <iostream>

auto main(int argc, char** argv) -> int {
#if defined(MPI_ENABLED)
  MPI_Init(&argc, &argv);
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

  std::cout << "Hello world from processor rank %d \n" << mpi_rank << "\n";

  Kokkos::initialize(argc, argv);
  {}
  Kokkos::finalize();

#if defined(MPI_ENABLED)

  MPI_Finalize();

#endif

  return 0;
}
