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

template <class ExecSpace>
struct SpaceInstance {
  static ExecSpace create() {
    return ExecSpace();
  }
  static void destroy(ExecSpace&) {}
  static bool overlap() {
    return false;
  }
};

#ifdef Kokkos_ENABLE_CUDA
template <>
struct SpaceInstance<Kokkos::Cuda> {
  static Kokkos::Cuda create() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return Kokkos::Cuda(stream);
  }
  static void destroy(Kokkos::Cuda& space) {
    cudaStream_t stream = space.cuda_stream();
    cudaStreamDestroy(stream);
  }
  static bool overlap() {
    bool value          = true;
    auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (local_rank_str) {
      value = (std::stoi(local_rank_str) == 0);
    }
    return value;
  }
};
#endif

// Initialize physical system

struct Init_kernel {
  Init_kernel(const Kokkos::View<double**>& T,
              const double&                 T1,
              const double&                 T0,
              const int&                    nx,
              const int&                    ny,
              const int&                    xdown,
              const int&                    ydown,
              const int&                    nghost)
    : m_T(T),
      m_T1(T1),
      m_T0(T0),
      m_nx(nx),
      m_ny(ny),
      m_xdown(xdown),
      m_ydown(ydown),
      m_nghost(nghost) {}

  KOKKOS_INLINE_FUNCTION void operator()(int i, int j) const {
    m_T(i, j) += (m_T1 - m_T0)
                 * math::exp(-((m_xdown + i - 0.5 * (m_nx + 2 * m_nghost))
                                 * (m_xdown + i - 0.5 * (m_nx + 2 * m_nghost))
                               + (m_ydown + j - 0.5 * (m_ny + 2 * m_nghost))
                                   * (m_ydown + j - 0.5 * (m_ny + 2 * m_nghost)))
                             / (0.2 * m_nx) / (0.2 * m_nx));
  }

private:
  Kokkos::View<double**> m_T;
  double                 m_T1, m_T0;
  int                    m_nx, m_ny, m_xdown, m_ydown, m_nghost;
};

struct CommHelper {
  MPI_Comm comm;

  int      mx, my;
  int      rank;
  int      xint, yint;
  int      up, down, left, right;

  CommHelper(MPI_Comm comm_) {
    comm = comm_;
    int nranks;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &rank);

    mx    = 2;
    my    = 1;
    yint  = floor(rank / mx);
    xint  = rank - yint * mx;
    left  = xint == 0 ? 0 + (mx - 1) + mx * yint : rank - 1;
    right = xint == mx - 1 ? 0 + 0 + mx * yint : rank + 1;
    down  = yint == 0 ? 0 + xint + mx * (my - 1) : rank - mx;
    up    = yint == my - 1 ? 0 + xint + mx * 0 : rank + mx;

    printf("#Ranks: %i This rank: %i nx/ny: %i %i This nx/ny: %i %i \n",
           nranks,
           rank,
           mx,
           my,
           xint,
           yint);
    printf("This rank: %i Neighbors (left,right,down,up): %i %i %i %i\n",
           rank,
           left,
           right,
           down,
           up);
  }

  template <class ViewType>
  void isend_irecv(int          partner,
                   ViewType     send_buffer,
                   ViewType     recv_buffer,
                   int          tag,
                   MPI_Request* request_send,
                   MPI_Request* request_recv) {
    MPI_Irecv(
      recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, tag, comm, request_recv);
    MPI_Isend(
      send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, tag, comm, request_send);
  }
};

struct System {
  CommHelper             comm;
  const int              nx = 1000, ny = 1000, nghost = 1;    // System size in grid points
  int                    sx, sy, imin, imax, jmin, jmax, lx, ly;
  int                    tmax, iout;    // Number of timesteps, output interval
  Kokkos::View<double**> T, Ti, dT;     // Fields of physical variables
  int                    xdown = 0, ydown = 0;
  int                    xup = 0, yup = 0;
  Kokkos::View<double**> io_recast;
  double                 T0, T1, vx, vy;       // Physical constants
  double                 dt;                   // Integration time-step
  double                 eloc, etot, etot0;    // Total energy
  using buffer_t = Kokkos::View<double*>;      // recall declaration
                                               // can include
                                               // Kokkos::CudaSpace
  buffer_t                      T_left, T_right, T_up, T_down;
  buffer_t                      T_left_out, T_right_out, T_up_out, T_down_out;
  int                           mpi_active_requests = 0;
  MPI_Request                   mpi_requests_recv[4];
  MPI_Request                   mpi_requests_send[4];

  Kokkos::DefaultExecutionSpace E_left
    = SpaceInstance<Kokkos::DefaultExecutionSpace>::create();
  Kokkos::DefaultExecutionSpace E_right
    = SpaceInstance<Kokkos::DefaultExecutionSpace>::create();
  Kokkos::DefaultExecutionSpace E_down
    = SpaceInstance<Kokkos::DefaultExecutionSpace>::create();
  Kokkos::DefaultExecutionSpace E_up = SpaceInstance<Kokkos::DefaultExecutionSpace>::create();

  // Specify mesh and physical constants
  System(MPI_Comm comm_) : comm(comm_) {
    tmax = 2000;
    T0 = 0.0, T1 = 1.0, vx = 0.5, vy = 0.0;
    dt = 0.5, iout = 20;
    Kokkos::deep_copy(T, T0);
    Kokkos::deep_copy(Ti, T0);
  }

  void setup_subdomain() {
    lx    = nx / comm.mx;
    xdown = lx * comm.xint;
    xup   = xdown + lx;
    if (xup > nx)
      xup = nx;
    ly    = ny / comm.my;
    ydown = ly * comm.yint;
    yup   = ydown + ly;
    if (yup > ny)
      yup = ny;

    printf("My Domain: %i (%i %i) (%i %i)\n", comm.rank, xdown, xup, ydown, yup);

    sx = lx + 2 * nghost, sy = ly + 2 * nghost;
    imin = nghost, imax = nghost + lx;
    jmin = nghost, jmax = nghost + ly;

    T           = Kokkos::View<double**>("System::T", sx, sy);
    Ti          = Kokkos::View<double**>("System::Ti", sx, sy);
    dT          = Kokkos::View<double**>("System::dT", sx, sy);
    io_recast   = Kokkos::View<double**>("System::io_recast", lx, ly);

    // incoming halos
    T_left      = buffer_t("System::T_left", sy);
    T_right     = buffer_t("System::T_right", sy);
    T_down      = buffer_t("System::T_down", sx);
    T_up        = buffer_t("System::T_up", sx);

    // outgoing halo
    T_left_out  = buffer_t("System::T_left_out", sy);
    T_right_out = buffer_t("System::T_right_out", sy);
    T_down_out  = buffer_t("System::T_down_out", sx);
    T_up_out    = buffer_t("System::T_up_out", sx);

    std::ostringstream msg;
    msg << "MPI rank(" << comm.rank << ") ";
    msg << "{" << std::endl;

    if (Kokkos::hwloc::available()) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
          << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
          << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
    }

    Kokkos::print_configuration(msg);

    msg << "}" << std::endl;

    std::cout << msg.str();
  }

  void pack_T_halo() {
    auto T_             = this->T;
    auto T_left_out_    = this->T_left_out;
    auto T_down_out_    = this->T_down_out;
    auto T_right_out_   = this->T_right_out;
    auto T_up_out_      = this->T_up_out;
    auto lx_            = this->lx;
    auto ly_            = this->ly;

    mpi_active_requests = 0;
    int mar             = 0;
    Kokkos::deep_copy(E_left, T_left_out_, Kokkos::subview(T_, 1, Kokkos::ALL));
    mar++;

    Kokkos::deep_copy(E_down, T_down_out_, Kokkos::subview(T_, Kokkos::ALL, 1));
    mar++;

    Kokkos::deep_copy(E_right, T_right_out_, Kokkos::subview(T_, lx_, Kokkos::ALL));
    mar++;

    Kokkos::deep_copy(E_up, T_up_out_, Kokkos::subview(T_, Kokkos::ALL, ly_));
    mar++;
  }

  void unpack_T_halo() {
    auto T_        = this->T;
    auto T_left_   = this->T_left;
    auto T_down_   = this->T_down;
    auto T_right_  = this->T_right;
    auto T_up_     = this->T_up;
    auto sx_       = this->sx;
    auto sy_       = this->sy;
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    Kokkos::parallel_for(
      "unpack", policy_t({ 0, 0 }, { sx_, sy_ }), KOKKOS_LAMBDA(int i, int j) {
        if (i == 0)
          T_(i, j) = T_left_(j);
        if (j == 0)
          T_(i, j) = T_down_(i);
        if (i == sx_ - 1)
          T_(i, j) = T_right_(j);
        if (j == sy_ - 1)
          T_(i, j) = T_up_(i);
      });
  }

  void exchange_T_halo() {
    auto T_left_out_  = this->T_left_out;
    auto T_down_out_  = this->T_down_out;
    auto T_right_out_ = this->T_right_out;
    auto T_up_out_    = this->T_up_out;
    auto T_left_      = this->T_left;
    auto T_down_      = this->T_down;
    auto T_right_     = this->T_right;
    auto T_up_        = this->T_up;
    auto xdown_       = this->xdown;
    auto xup_         = this->xup;
    auto ydown_       = this->ydown;
    auto yup_         = this->yup;
    auto nx_          = this->nx;
    auto ny_          = this->ny;

    int  mar          = 0;
    int  tag;

    tag = 0;
    if (xdown_ == 0)
      tag = 1;
    auto outvec0 = Kokkos::create_mirror_view(T_left_out_);
    auto invec0  = Kokkos::create_mirror_view(T_left_);
    Kokkos::deep_copy(E_left, outvec0, T_left_out_);
    E_left.fence();
    comm.isend_irecv(
      comm.left, outvec0, invec0, tag, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
    mar++;

    tag = 0;
    if (ydown_ == 0)
      tag = 1;
    auto outvec1 = Kokkos::create_mirror_view(T_down_out_);
    auto invec1  = Kokkos::create_mirror_view(T_down_);
    Kokkos::deep_copy(E_down, outvec1, T_down_out_);
    E_down.fence();
    comm.isend_irecv(
      comm.down, outvec1, invec1, tag, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
    mar++;

    tag = 0;
    if (xup_ == nx_)
      tag = 1;
    auto outvec2 = Kokkos::create_mirror_view(T_right_out_);
    auto invec2  = Kokkos::create_mirror_view(T_right_);
    Kokkos::deep_copy(E_right, outvec2, T_right_out_);
    E_right.fence();
    comm.isend_irecv(
      comm.right, outvec2, invec2, tag, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
    mar++;

    tag = 0;
    if (yup_ == ny_)
      tag = 1;
    auto outvec3 = Kokkos::create_mirror_view(T_up_out_);
    auto invec3  = Kokkos::create_mirror_view(T_up_);
    Kokkos::deep_copy(E_up, outvec3, T_up_out_);
    E_up.fence();
    comm.isend_irecv(
      comm.up, outvec3, invec3, tag, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
    mar++;
    mpi_active_requests = mar;

    if (mpi_active_requests > 0) {
      MPI_Waitall(mpi_active_requests, mpi_requests_send, MPI_STATUSES_IGNORE);
      MPI_Waitall(mpi_active_requests, mpi_requests_recv, MPI_STATUSES_IGNORE);
    }

    Kokkos::deep_copy(E_left, T_left_, invec0);
    Kokkos::deep_copy(E_down, T_down_, invec1);
    Kokkos::deep_copy(E_right, T_right_, invec2);
    Kokkos::deep_copy(E_up, T_up_, invec3);

  }

  void pack_Ti_halo() {
    auto Ti_            = this->Ti;
    auto T_left_out_    = this->T_left_out;
    auto T_down_out_    = this->T_down_out;
    auto T_right_out_   = this->T_right_out;
    auto T_up_out_      = this->T_up_out;
    auto lx_            = this->lx;
    auto ly_            = this->ly;

    mpi_active_requests = 0;
    int mar             = 0;
    Kokkos::deep_copy(E_left, T_left_out_, Kokkos::subview(Ti_, 1, Kokkos::ALL));
    mar++;

    Kokkos::deep_copy(E_down, T_down_out_, Kokkos::subview(Ti_, Kokkos::ALL, 1));
    mar++;

    Kokkos::deep_copy(E_right, T_right_out_, Kokkos::subview(Ti_, lx_, Kokkos::ALL));
    mar++;

    Kokkos::deep_copy(E_up, T_up_out_, Kokkos::subview(Ti_, Kokkos::ALL, ly_));
    mar++;
  }

  void unpack_Ti_halo() {
    auto Ti_       = this->Ti;
    auto T_left_   = this->T_left;
    auto T_down_   = this->T_down;
    auto T_right_  = this->T_right;
    auto T_up_     = this->T_up;
    auto sx_       = this->sx;
    auto sy_       = this->sy;
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    Kokkos::parallel_for(
      "unpack", policy_t({ 0, 0 }, { sx_, sy_ }), KOKKOS_LAMBDA(int i, int j) {
        if (i == 0)
          Ti_(i, j) = T_left_(j);
        if (j == 0)
          Ti_(i, j) = T_down_(i);
        if (i == sx_ - 1)
          Ti_(i, j) = T_right_(j);
        if (j == sy_ - 1)
          Ti_(i, j) = T_up_(i);
      });
  }

  void initialize() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    Kokkos::parallel_for("initial",
                         policy_t({ imin, jmin }, { imax, jmax }),
                         Init_kernel(T, T1, T0, nx, ny, xdown, ydown, nghost));
  }

  // Advance physical time
  void evolve(MPI_Comm comm_) {
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
    auto lx_        = this->lx;
    auto ly_        = this->ly;
    auto nghost_    = this->nghost;
    auto eloc_      = this->eloc;
    auto etot_      = this->etot;
    auto etot0_     = this->etot0;
    auto imin_      = this->imin;
    auto imax_      = this->imax;
    auto jmin_      = this->jmin;
    auto jmax_      = this->jmax;
    auto xdown_     = this->xdown;
    auto xup_       = this->xup;
    auto ydown_     = this->ydown;
    auto yup_       = this->yup;

#ifdef OUTPUT_ENABLED
    adios2::ADIOS      adios(comm_);
    // adios2::ADIOS      adios;
    adios2::IO         io = adios.DeclareIO("fieldIO");
    const adios2::Dims shape { static_cast<std::size_t>(nx), static_cast<std::size_t>(ny) };
    const adios2::Dims start { static_cast<std::size_t>(xdown_),
                               static_cast<std::size_t>(ydown_) };
    const adios2::Dims count { static_cast<std::size_t>(lx_), static_cast<std::size_t>(ly_) };
    auto               io_variable = io.DefineVariable<double>("data", shape, start, count);
    io.SetEngine("HDF5");

    adios2::Engine adios_engine = io.Open("../Temp.h5", adios2::Mode::Write);
#endif

    double time_push = 0.0, time_dump = 0.0, time_bnd = 0.0, time_tot = 0.0;

    Kokkos::parallel_reduce(
      policy_t({ imin_, jmin_ }, { imax_, jmax_ }),
      KOKKOS_LAMBDA(int i, int j, double& lval) { lval += 0.5 * T_(i, j) * T_(i, j); },
      eloc_);

    MPI_Reduce(&eloc_, &etot0_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    for (int t = 0; t <= tmax; t++) {
      time_push -= timer.seconds();
      time_tot -= timer.seconds();

      //  Do a Crank-Nicolson stage
      Kokkos::parallel_for(
        "CN1", policy_t({ imin_, jmin_ }, { imax_, jmax_ }), KOKKOS_LAMBDA(int i, int j) {
          dT_(i, j) = 0.5 * vx_ * (T_(i + 1, j) - T_(i - 1, j));
          dT_(i, j) += 0.5 * vy_ * (T_(i, j + 1) - T_(i, j - 1));
        });

      Kokkos::parallel_for(
        "Euler1", policy_t({ imin_, jmin_ }, { imax_, jmax_ }), KOKKOS_LAMBDA(int i, int j) {
          Ti_(i, j) = T_(i, j) + 0.5 * dt_ * dT_(i, j);
        });

      time_push += timer.seconds();

      time_bnd -= timer.seconds();

      pack_Ti_halo();
      exchange_T_halo();
      unpack_Ti_halo();

      time_bnd += timer.seconds();

      time_push -= timer.seconds();

      //  Do final Crank-Nicolson stage
      Kokkos::parallel_for(
        "CN2", policy_t({ imin_, jmin_ }, { imax_, jmax_ }), KOKKOS_LAMBDA(int i, int j) {
          dT_(i, j) = 0.5 * vx_ * (Ti_(i + 1, j) - Ti_(i - 1, j));
          dT_(i, j) += 0.5 * vy_ * (Ti_(i, j + 1) - Ti_(i, j - 1));
        });

      Kokkos::parallel_for(
        "Euler2", policy_t({ imin_, jmin_ }, { imax_, jmax_ }), KOKKOS_LAMBDA(int i, int j) {
          T_(i, j) += dt_ * dT_(i, j);
        });

      time_push += timer.seconds();

      time_bnd -= timer.seconds();

      pack_T_halo();
      exchange_T_halo();
      unpack_T_halo();

      time_bnd += timer.seconds();

      time_dump -= timer.seconds();

        Kokkos::parallel_reduce(
          policy_t({ imin_, jmin_ }, { imax_, jmax_ }),
          KOKKOS_LAMBDA(int i, int j, double& lval) { lval += 0.5 * T_(i, j) * T_(i, j); },
          eloc_);

        MPI_Reduce(&eloc_, &etot_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      time_dump += timer.seconds();

      if (t % iout == 0 || t == tmax) {
        time_dump -= timer.seconds();

#ifdef OUTPUT_ENABLED

        Kokkos::parallel_for(
          "recast", policy_t({ 0, 0 }, { lx_, ly_ }), KOKKOS_LAMBDA(int i, int j) {
            io_recast_(i, j) = T_(i + nghost_, j + nghost_);
          });

        auto outvec = Kokkos::create_mirror_view(io_recast_);
        Kokkos::deep_copy(outvec, io_recast_);

        adios_engine.BeginStep();
        adios_engine.Put<double>(io_variable, outvec);
        adios_engine.EndStep();

#endif

        time_dump += timer.seconds();
      }

      time_tot += timer.seconds();

      if (comm.rank == 0) {
        printf("Timestep %i Timing (Wall: %lf; Total: %lf; Average: %lf)\n",
               t,
               timer.seconds(),
               time_tot,
               time_tot / t);
        printf("Pusher: Timing (Total: %lf; %.2f%%)\n",
               time_push / t,
               time_push / time_tot * 100.0);
        printf("Boundary: Timing (Total: %lf; %.2f%%)\n",
               time_bnd / t,
               time_bnd / time_tot * 100.0);
        printf("Output: Timing (Total: %lf; %.2f%%)\n",
               time_dump / t,
               time_dump / time_tot * 100.0);
        printf("Energy difference: %.2f%%\n", 100.0 * (etot_ - etot0_) / etot0_);
        printf("\n");
      }
    }

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
    System sys(MPI_COMM_WORLD);
    sys.setup_subdomain();
    sys.initialize();
    sys.evolve(MPI_COMM_WORLD);
  }
  Kokkos::finalize();

#ifdef MPI_ENABLED
  MPI_Finalize();
#endif

  return 0;
}
