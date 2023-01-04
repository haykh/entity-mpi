#include <Kokkos_Core.hpp>
#include <iostream>
#include <adios2.h>
#if ADIOS2_USE_MPI
#include <mpi.h>
#endif


auto main(int argc, char* argv[]) -> int {
  
  Kokkos::initialize(argc, argv);
  {

  // Initialize mesh

  // Advance physical time

  // Output data

  }
  Kokkos::finalize();

  return 0;
}
