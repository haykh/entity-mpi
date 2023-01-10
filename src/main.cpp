#include <Kokkos_Core.hpp>

#include <iostream>

#ifdef MPI_ENABLED
#  include <mpi.h>
#endif

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
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
  const int nx = 100, ny = 100, nghost = 2;                // System size in grid points
  const int sx = nx + 2 * nghost, sy = ny + 2 * nghost;    // System size including ghost cells
  const int imin = nghost, imax = nghost + nx;
  const int jmin = nghost, jmax = nghost + ny;
  int       tmax, iout;                // Number of timesteps, output interval
  Kokkos::View<double**> T, Ti, dT;    // Fields of physical variables
  Kokkos::View<double**> io_recast;
  double                 T0, T1, vx, vy;    // Physical constants
  double                 dt;                // Integration time-step
  double                 etot;              // Total energy

  // Specify mesh and physical constants
  System()
    : T("System::T", sx, sy),
      Ti("System::Ti", sx, sy),
      dT("System::dT", sx, sy),
      io_recast("System::io_recast", nx, ny) {
    tmax = 4000;
    T0 = 0.0, T1 = 1.0, vx = 0.5, vy = 0.0;
    dt = 0.5, iout = 40;
    Kokkos::deep_copy(T, T0);
    Kokkos::deep_copy(Ti, T0);
  }

  void initialize() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    Kokkos::parallel_for("initial",
                         policy_t({ imin, jmin }, { imax, jmax }),
                         Init_kernel(T, T1, T0, nx, ny, nghost));
  }

  // Advance physical time
  void evolve() {
    Kokkos::Timer timer;
    using policy_t  = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    auto dT_        = this->dT;
    auto T_         = this->T;
    auto Ti_        = this->Ti;
    auto io_recast_ = this->io_recast;
    auto vx_        = this->vx;
    auto vy_        = this->vy;
    auto dt_        = this->dt;
    auto nx_        = this->nx;
    auto ny_        = this->ny;
    auto nghost_    = this->nghost;
    auto etot_      = this->etot;
    // HH: you only need to use these references in the LAMBDA executed on the device

#ifdef OUTPUT_ENABLED
    adios2::ADIOS      adios;
    adios2::IO         io = adios.DeclareIO("fieldIO");
    const adios2::Dims shape { static_cast<std::size_t>(sx), static_cast<std::size_t>(sy) };
    const adios2::Dims start { 0, 0 };
    const adios2::Dims count { static_cast<std::size_t>(sx), static_cast<std::size_t>(sy) };
    auto               io_variable = io.DefineVariable<double>("data", shape, start, count);
    // io.DefineAttribute<std::string>("unit", "K", "Temperature");
    io.SetEngine("HDF5");

    adios2::Engine adios_engine = io.Open("Temp.h5", adios2::Mode::Write);
#endif

    // for (int t = 0; t <= tmax; t++) {

    double time = timer.seconds();

    // //  Do a Crank-Nicolson stage
    // Kokkos::parallel_for(
    //   "CN1",
    //   policy_t({ nghost_, nghost_ }, { nx_ + nghost_, ny_ + nghost_ }),
    //   KOKKOS_LAMBDA(int i, int j) {
    //     dT_(i, j) = 0.5 * vx_ * (T_(i + 1, j) - T_(i - 1, j));
    //     dT_(i, j) += 0.5 * vy_ * (T_(i, j + 1) - T_(i, j - 1));
    //   });

    // Kokkos::parallel_for(
    //   "Euler1",
    //   policy_t({ nghost_, nghost_ }, { nx_ + nghost_, ny_ + nghost_ }),
    //   KOKKOS_LAMBDA(int i, int j) { Ti_(i, j) = T_(i, j) + 0.5 * dt_ * dT_(i, j); });

    // //  Exchange halo for periodic boundary
    // Kokkos::parallel_for(
    //   "Boundary2",
    //   policy_t({ nghost_, 0 }, { nx_ + nghost_, nghost_ }),
    //   KOKKOS_LAMBDA(int i, int j) {
    //     Ti_(i, j)               = Ti_(i, ny_ + j);
    //     Ti_(i, ny_ + nghost_ + j) = Ti_(i, nghost_ + j);
    //     Ti_(j, i)               = Ti_(nx_ + j, i);
    //     Ti_(nx_ + nghost_ + j, i) = Ti_(nghost_ + j, i);
    //   });

    // //  Do final Crank-Nicolson stage
    // Kokkos::parallel_for(
    //   "CN2",
    //   policy_t({ nghost_, nghost_ }, { nx_ + nghost_, ny_ + nghost_ }),
    //   KOKKOS_LAMBDA(int i, int j) {
    //     dT_(i, j) = 0.5 * vx_ * (Ti_(i + 1, j) - Ti_(i - 1, j));
    //     dT_(i, j) += 0.5 * vy_ * (Ti_(i, j + 1) - Ti_(i, j - 1));
    //   });

    // Kokkos::parallel_for(
    //   "Euler2",
    //   policy_t({ nghost_, nghost_ }, { nx_ + nghost_, ny_ + nghost_ }),
    //   KOKKOS_LAMBDA(int i, int j) { T_(i, j) += dt_ * dT_(i, j); });

    // //  Exchange halo for periodic boundary
    // Kokkos::parallel_for(
    //   "Boundary2",
    //   policy_t({ nghost_, 0 }, { nx_ + nghost_, nghost_ }),
    //   KOKKOS_LAMBDA(int i, int j) {
    //     T_(i, j)               = T_(i, ny_ + j);
    //     T_(i, ny_ + nghost_ + j) = T_(i, nghost_ + j);
    //     T_(j, i)               = T_(nx_ + j, i);
    //     T_(nx_ + nghost_ + j, i) = T_(nghost_ + j, i);
    //   });

    // if (t % iout == 0 || t == tmax) {

    // etot_ = 0.0;
    // Kokkos::parallel_reduce(policy_t({ nghost_, nghost_ }, { nx_ + nghost_, ny_ + nghost_
    // }),
    //   KOKKOS_LAMBDA (int i, int j, double& lval) {lval += 0.5*T_(i, j)*T_(i, j);},etot_);

#ifdef OUTPUT_ENABLED

    // Kokkos::parallel_for(
    //   "recast", policy_t({ 0, 0 }, { nx_, ny_ }), KOKKOS_LAMBDA(int i, int j) {
    //     io_recast_(i, j) = T_(i + nghost_, j + nghost_);
    //   });

    adios2::Box<adios2::Dims> sel(
      { 0, 0 }, { static_cast<std::size_t>(sx), static_cast<std::size_t>(sy) });
    io_variable.SetSelection(sel);

    adios_engine.BeginStep();
    // adios_engine.Put<double>(io_variable, Kokkos::subview(T, Kokkos::ALL(), Kokkos::ALL()));
    auto T_h = Kokkos::create_mirror_view(T);
    adios_engine.Put(io_variable, T_h);
    adios_engine.EndStep();

    // adios_engine.Close();
    // adios_engine = io.Open("../Temp.h5", adios2::Mode::Append);

#endif

    // printf("Timestep %i Timing (Total: %lf; Average: %lf; Energy: %lf)\n", t, time, time /
    // t, etot_);
    // }
    // }

#ifdef OUTPUT_ENABLED
    adios_engine.Close();
#endif
  }
};

auto main(int argc, char* argv[]) -> int {
#ifdef MPI_ENABLED
  MPI_Init(&argc, &argv);
#endif

  Kokkos::initialize(argc, argv);
  {
    System sys;
    // sys.initialize();
    sys.evolve();
  }
  Kokkos::finalize();

#ifdef MPI_ENABLED
  MPI_Finalize();
#endif

  return 0;
}
