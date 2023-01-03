#include <Kokkos_Core.hpp>

#include <iostream>

#include <adios2.h>
#if ADIOS2_USE_MPI
#include <mpi.h>
#endif


auto main(int argc, char* argv[]) -> int {
  
  int N = 10;
  double result;

  Kokkos::initialize(argc, argv);
  {

  Kokkos::View<double*> y( "y", N );

  Kokkos::parallel_for( "y_init", N, KOKKOS_LAMBDA ( int i ) {
    y( i ) = 1;
  });

  Kokkos::parallel_for( "y_add", N, KOKKOS_LAMBDA ( int i ) {
    y( i ) += 0.1;
  });
  
  Kokkos::parallel_reduce("Loop1", N, KOKKOS_LAMBDA (const int& i, double& lsum ) {
    lsum += y( i );
  },result);

  std::cout << result << std::endl;

  }
  Kokkos::finalize();

  return 0;
}
