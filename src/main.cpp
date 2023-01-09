#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <iostream>
#if ADIOS2_USE_MPI
#  include <mpi.h>
#endif

namespace math = Kokkos;

// Initialize physical system

struct Init_kernel {
  Init_kernel(const Kokkos::View<double**>& T,
              const double&                 T1,
              const double&                 T0,
              const int&                    nx,
              const int&                    ny,
              const int&                    nghost)
    : m_T(T), m_T1(T1), m_T0(T0), m_nx(nx), m_ny(ny), m_nghost(nghost) {}

  KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const {
    m_T(i, j)
      += (m_T1 - m_T0)
         * math::exp(-((i - 0.5 * (m_nx + 2 * m_nghost)) * (i - 0.5 * (m_nx + 2 * m_nghost))
                       + (j - 0.5 * (m_ny + 2 * m_nghost)) * (j - 0.5 * (m_ny + 2 * m_nghost)))
                     / (0.2 * m_nx) / (0.2 * m_nx));
  }

private:
  Kokkos::View<double**> m_T;
  double                 m_T1, m_T0;
  int                    m_nx, m_ny, m_nghost;
};

struct System {
  int                    nx = 100, ny = 100, nghost = 2;    // System size in grid points
  int                    tmax, iout;        // Number of timesteps, output interval
  Kokkos::View<double**> T, Ti, dT;         // Fields of physical variables
  Kokkos::View<double**> io_recast;
  double                 T0, T1, vx, vy;    // Physical constants
  double                 dt;                // Integration time-step

  // Specify mesh and physical constants
  System() : T ("System::T", nx + 2 * nghost, ny + 2 * nghost), 
             Ti ("System::Ti", nx + 2 * nghost, ny + 2 * nghost), 
             dT ("System::dT", nx + 2 * nghost, ny + 2 * nghost), 
             io_recast ("System::io_recast", nx, ny)
  {
    tmax      = 4000;
    T0 = 0.0, T1 = 1.0, vx = 0.5, vy = 0.0;
    dt = 0.5, iout = 40;
    Kokkos::deep_copy(T, T0);
    Kokkos::deep_copy(Ti, T0);
  }

  void initialize() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    Kokkos::parallel_for("recast",
                         policy_t({ nghost, nghost }, { nx + nghost, ny + nghost }),
                         Init_kernel(T, T1, T0, nx, ny, nghost));
  }

  // Advance physical time
  void evolve() {
    Kokkos::Timer timer;
    // adios2::ADIOS adios;
    // adios2::IO    io          = adios.DeclareIO("fieldIO");
    adios2::Dims count;
    count.push_back(nx);
    count.push_back(ny);
    // auto          io_variable = io.DefineVariable<double>("Temperature", {}, {}, count);
    // io.DefineAttribute<std::string>("unit", "K", "Temperature");
    // io.SetEngine("HDF5");

    // adios2::Engine adios_engine = io.Open("../Temp.h5");
    using policy_t              = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    for (int t = 0; t <= tmax; t++) {

      double time = timer.seconds();

      auto dT_ = this->dT;

      //  Do a Crank-Nicolson stage
      Kokkos::parallel_for(
        "CN1",
        policy_t({ nghost, nghost }, { nx + nghost, ny + nghost }),
        KOKKOS_LAMBDA(int i, int j) {
          dT_(i, j) = 0.5 * vx;// * (T(i + 1, j) - T(i - 1, j));
          // dT(i, j) += 0.5 * vy * (T(i, j + 1) - T(i, j - 1));
        });

      // Kokkos::parallel_for(
      //   "Euler1",
      //   policy_t({ nghost, nghost }, { nx + nghost, ny + nghost }),
      //   KOKKOS_LAMBDA(int i, int j) { Ti(i, j) = T(i, j) + 0.5 * dt * dT(i, j); });

      // //  Exchange halo for periodic boundary
      // Kokkos::parallel_for(
      //   "Boundary2",
      //   policy_t({ nghost, 0 }, { nx + nghost, nghost }),
      //   KOKKOS_LAMBDA(int i, int j) {
      //     Ti(i, j)               = Ti(i, ny + j);
      //     Ti(i, ny + nghost + j) = Ti(i, nghost + j);
      //     Ti(j, i)               = Ti(nx + j, i);
      //     Ti(nx + nghost + j, i) = Ti(nghost + j, i);
      //   });

      // //  Do final Crank-Nicolson stage
      // Kokkos::parallel_for(
      //   "CN2",
      //   policy_t({ nghost, nghost }, { nx + nghost, ny + nghost }),
      //   KOKKOS_LAMBDA(int i, int j) {
      //     dT(i, j) = 0.5 * vx * (Ti(i + 1, j) - Ti(i - 1, j));
      //     dT(i, j) += 0.5 * vy * (Ti(i, j + 1) - Ti(i, j - 1));
      //   });

      // Kokkos::parallel_for(
      //   "Euler2",
      //   policy_t({ nghost, nghost }, { nx + nghost, ny + nghost }),
      //   KOKKOS_LAMBDA(int i, int j) { T(i, j) += dt * dT(i, j); });

      // //  Exchange halo for periodic boundary
      // Kokkos::parallel_for(
      //   "Boundary2",
      //   policy_t({ nghost, 0 }, { nx + nghost, nghost }),
      //   KOKKOS_LAMBDA(int i, int j) {
      //     T(i, j)               = T(i, ny + j);
      //     T(i, ny + nghost + j) = T(i, nghost + j);
      //     T(j, i)               = T(nx + j, i);
      //     T(nx + nghost + j, i) = T(nghost + j, i);
      //   });

      if (t % iout == 0 || t == tmax) {

        // Kokkos::parallel_for(
        //   "recast", policy_t({ 0, 0 }, { nx, ny }), KOKKOS_LAMBDA(int i, int j) {
        //     io_recast(i, j) = T(i + nghost, j + nghost);
        //   });

        // adios_engine.BeginStep();
        // adios_engine.Put(io_variable, io_recast, adios2::Mode::Deferred);
        // adios_engine.EndStep();

        printf("Timestep %i Timing (Total: %lf; Average: %lf)\n", t, time, time / t);
      }
    }

    // adios_engine.Close();
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
