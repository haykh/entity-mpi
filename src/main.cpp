#include <Kokkos_Core.hpp>
#include <iostream>
#include <adios2.h>
#if ADIOS2_USE_MPI
#include <mpi.h>
#endif

// Initialize physical system

struct System {

  int nx, ny, nz;                 // System size in grid points
  int tmax;                       // Number of timesteps
  Kokkos::View<double***> T, dT;  // Fields of physical variables
  double T0, q, sigma, P;         // Physical constants
  double dt;                      // Integration time-step

  System() {
    nx = 200, ny = 200, nz = 200;
    tmax = 100;
    T = Kokkos::View<double***>(), dT = Kokkos::View<double***>();
    T0 = 0.0, q = 1.0, sigma = 1.0, P = 1.0;
    dt = 0.1; 
  }

};

  // Advance physical time

  // Output data

auto main(int argc, char* argv[]) -> int {

  Kokkos::initialize(argc, argv);
  {
    System sys;
  }
  Kokkos::finalize();

  return 0;
}


  // Kokkos::View<double*> y( "y", N );

  // Kokkos::parallel_for( "y_init", N, KOKKOS_LAMBDA ( int i ) {
  //   y( i ) = 1;
  // });

  // Kokkos::parallel_for( "y_add", N, KOKKOS_LAMBDA ( int i ) {
  //   y( i ) += 0.1;
  // });
  
  // Kokkos::parallel_reduce("Loop1", N, KOKKOS_LAMBDA (const int& i, double& lsum ) {
  //   lsum += y( i );
  // },result);

