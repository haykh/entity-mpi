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
    nx = 20, ny = 20, nz = 20;
    tmax = 100;
    T = Kokkos::View<double***>("System::T", nx, ny, nz);
    dT = Kokkos::View<double***>("System::dT", nx, ny, nz);
    T0 = 0.0, q = 1.0, sigma = 1.0, P = 1.0;
    dt = 0.1; 
    Kokkos::deep_copy(T,T0);
  }

  void output() {
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("fieldIO");
    auto io_variable = io.DefineVariable<double>(
        "Temperature", {size_t(nx),size_t(ny),size_t(nz)}, {0,0,0}, {size_t(nx),size_t(ny),size_t(nz)}, adios2::ConstantDims);
    io.DefineAttribute<std::string>("unit", "K", "Temperature");
    io.SetEngine("HDF5");
    
    adios2::Engine adios_engine = io.Open("../Temp.h5", adios2::Mode::Write);
    adios_engine.BeginStep();
    adios_engine.Put(io_variable, T.data());
    adios_engine.EndStep();
    adios_engine.Close();
  }

};

  // Advance physical time

  // Output data

auto main(int argc, char* argv[]) -> int {

  Kokkos::initialize(argc, argv);
  {
    System sys;
    sys.output();
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

