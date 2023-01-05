#include <Kokkos_Core.hpp>
#include <iostream>
#include <adios2.h>
#if ADIOS2_USE_MPI
#include <mpi.h>
#endif

// Initialize physical system

struct System {

  int nx, ny, nghost;                 // System size in grid points
  int tmax, iout;                     // Number of timesteps, output interval
  Kokkos::View<double**> T, Ti, dT;   // Fields of physical variables
  Kokkos::View<double**> io_recast;
  double T0, T1, vx, vy;                  // Physical constants
  double dt;                          // Integration time-step

  // Specify mesh and physical constants
  System() {
    nx = 20, ny = 20, nghost = 2;
    tmax = 100;
    T = Kokkos::View<double**>("System::T", nx+2*nghost, ny+2*nghost);
    Ti = Kokkos::View<double**>("System::Ti", nx+2*nghost, ny+2*nghost);
    dT = Kokkos::View<double**>("System::dT", nx+2*nghost, ny+2*nghost);
    io_recast = Kokkos::View<double**>("io_recast", nx, ny);
    T0 = 0.0, T1 = 1.0, vx = 0.5, vy = 0.0;
    dt = 0.1, iout = 10; 
    Kokkos::deep_copy(T,T0);
    Kokkos::deep_copy(Ti,T0);
  }

  void initialize() {

    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
      Kokkos::parallel_for( "recast", policy_t({nghost,nghost},{nx+nghost,ny+nghost}), KOKKOS_LAMBDA ( int i, int j ) {
        T(i,j) += (T1 - T0) * exp(- ((i - 0.5*(nx + 2*nghost))*(i - 0.5*(nx + 2*nghost)) + (j-0.5*(ny + 2*nghost))*(j-0.5*(ny + 2*nghost))) / (0.2*20) / (0.2*20));
      });

  }

  // Advance physical time
  void evolve() {
    Kokkos::Timer timer;
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("fieldIO");
    auto io_variable = io.DefineVariable<double>(
        "Temperature", {size_t(nx),size_t(ny)}, {0,0}, {size_t(nx),size_t(ny)}, adios2::ConstantDims);
    io.DefineAttribute<std::string>("unit", "K", "Temperature");
    io.SetEngine("HDF5");
    
    adios2::Engine adios_engine = io.Open("../Temp.h5", adios2::Mode::Write);
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
      
      for(int t = 0; t <= tmax; t++) {
      
      //  Do a Crank-Nicolson step
      Kokkos::parallel_for( "recast", policy_t({nghost,nghost},{nx+nghost,ny+nghost}), KOKKOS_LAMBDA ( int i, int j ) {
        dT(i,j)  = 0.5 * vx * (T(i+1, j)-T(i-1, j));
        dT(i,j) += 0.5 * vy * (T(i, j+1)-T(i, j-1));
      });

      Kokkos::parallel_for( "recast", policy_t({nghost,nghost},{nx+nghost,ny+nghost}), KOKKOS_LAMBDA ( int i, int j ) {
        Ti(i,j)  = T(i,j) + dt * dT(i,j);
      });

      Kokkos::parallel_for( "recast", policy_t({nghost,nghost},{nx+nghost,ny+nghost}), KOKKOS_LAMBDA ( int i, int j ) {
        dT(i,j)  = 0.5 * vx * (Ti(i+1, j)-Ti(i-1, j));
        dT(i,j) += 0.5 * vy * (Ti(i, j+1)-Ti(i, j-1));
      });

      Kokkos::parallel_for( "recast", policy_t({nghost,nghost},{nx+nghost,ny+nghost}), KOKKOS_LAMBDA ( int i, int j ) {
        T(i,j)  = T(i,j) + dt * dT(i,j);
      });

      if(t%iout == 0 || t == tmax) {
        double time = timer.seconds();

        Kokkos::parallel_for( "recast", policy_t({0,0},{nx,ny}), KOKKOS_LAMBDA ( int i, int j ) {
          io_recast(i,j) = T(i+nghost, j+nghost);
        });

        adios_engine.BeginStep();
        adios_engine.Put(io_variable, io_recast.data());
        adios_engine.EndStep();

        printf("Timestep %i Timing (Total: %lf; Average: %lf)\n",t,time,time/t);
      }

    }

    adios_engine.Close();

  }

};

auto main(int argc, char* argv[]) -> int {

  Kokkos::initialize(argc, argv);
  {
    System sys;
    sys.initialize();
    sys.evolve();
  }
  Kokkos::finalize();

  return 0;
}