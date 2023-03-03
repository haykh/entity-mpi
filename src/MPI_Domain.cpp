#include <Kokkos_Core.hpp>

#include <iostream>

#ifdef MPI_ENABLED
#  include <mpi.h>
#endif

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#define PATCH 0
#define RANK 1
#define XINT 2
#define YINT 3
#define LEFT 4
#define RIGHT 5
#define UP 6
#define DOWN 7
#define LX 8
#define LY 9
#define XDOWN 10
#define XUP 11
#define YDOWN 12
#define YUP 13
#define SX 14
#define SY 15
#define IMIN 16
#define IMAX 17
#define JMIN 18
#define JMAX 19

namespace math = Kokkos;

struct CommHelper {
  MPI_Comm comm;

  int      mx, my;
  int      np;
  int      rank;
  int      nrank;
  int      xint, yint;
  int      up, down, left, right;

  CommHelper(MPI_Comm comm_) {
    comm = comm_;
    MPI_Comm_size(comm, &nrank);
    MPI_Comm_rank(comm, &rank);

    mx    = 3;
    my    = 3;  

    np = 2;
    if (rank == 1) np = 4;
    if (rank == 2) np = 3;
    
    // printf("#Ranks: %i This rank: %i nx/ny: %i %i This nx/ny: %i %i \n",
    //        nranks,
    //        rank,
    //        mx,
    //        my,
    //        xint,
    //        yint);
    // printf("This rank: %i Neighbors (left,right,down,up): %i %i %i %i\n",
    //        rank,
    //        left,
    //        right,
    //        down,
    //        up);
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

  template <class ViewType>
  void isend(int          partner,
                   ViewType     send_buffer,
                   int          tag,
                   MPI_Request* request_send) {
    MPI_Isend(
      send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, tag, comm, request_send);
  }

  template <class ViewType>
  void irecv(int          partner,
                   ViewType     recv_buffer,
                   int          tag,
                   MPI_Request* request_recv) {
    MPI_Irecv(
      recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, tag, comm, request_recv);
  }

};

struct System {
  CommHelper             comm;
  const int              nx = 600, ny = 600, nghost = 1;    // System size in grid points
  int                    sx, sy, imin, imax, jmin, jmax, lx, ly;
  int                    tmax, iout;    // Number of timesteps, output interval
  Kokkos::View<double***> T, Ti, dT;     // Fields of physical variables
  int                    xdown = 0, ydown = 0;
  int                    xup = 0, yup = 0;
  Kokkos::View<double**> io_recast;
  Kokkos::View<int**> meshdomain;
  Kokkos::View<int*> wholeworld;
  double                 T0, T1, vx, vy;       // Physical constants
  double                 dt;                   // Integration time-step
  double                 eloc, etot, etot0;    // Total energy
  using buffer_t = Kokkos::View<double*>;      // recall declaration
                                               // can include
                                               // Kokkos::CudaSpace
  buffer_t                      T_left, T_right, T_up, T_down;
  buffer_t                      T_left_out, T_right_out, T_up_out, T_down_out;
  int                           mpi_active_requests = 0;

  // Specify mesh and physical constants
  System(MPI_Comm comm_) : comm(comm_) {
    tmax = 10000;
    T0 = 0.0, T1 = 1.0, vx = 0.5, vy = 0.0;
    dt = 0.5, iout = 100;
    Kokkos::deep_copy(T, T0);
    Kokkos::deep_copy(Ti, T0);
  }

  void setup_subdomain() {

    meshdomain = Kokkos::View<int**>("System::Mesh", 20, comm.np);
    wholeworld = Kokkos::View<int*>("System::World", 9); 

    wholeworld(0) = 0;
    wholeworld(1) = 0;
    wholeworld(2) = 1;
    wholeworld(3) = 1;
    wholeworld(4) = 1;
    wholeworld(5) = 1;
    wholeworld(6) = 2;
    wholeworld(7) = 2;
    wholeworld(8) = 2;

    for (int i = 0; i < comm.np; i++) {
      if(comm.rank == 0) meshdomain(PATCH, i) = i;
      if(comm.rank == 1) meshdomain(PATCH, i) = 2 + i;
      if(comm.rank == 2) meshdomain(PATCH, i) = 6 + i;
      meshdomain(RANK, i) = comm.rank;
    }

    for (int i = 0; i < comm.np; i++) {
      meshdomain(YINT, i)  = floor(meshdomain(PATCH, i) / comm.mx);
      meshdomain(XINT, i)  = meshdomain(PATCH, i) - meshdomain(YINT, i) * comm.mx;
      meshdomain(LEFT, i)  = meshdomain(XINT, i) == 0 ? 0 + (comm.mx - 1) + comm.mx * meshdomain(YINT, i) : meshdomain(PATCH, i) - 1;
      meshdomain(RIGHT, i) = meshdomain(XINT, i) == comm.mx - 1 ? 0 + 0 + comm.mx * meshdomain(YINT, i) : meshdomain(PATCH, i) + 1;
      meshdomain(DOWN, i)  = meshdomain(YINT, i) == 0 ? 0 + meshdomain(XINT, i) + comm.mx * (comm.my - 1) : meshdomain(PATCH, i) - comm.mx;
      meshdomain(UP, i)    = meshdomain(YINT, i) == comm.my - 1 ? 0 + meshdomain(XINT, i) + comm.mx * 0 : meshdomain(PATCH, i) + comm.mx;
      meshdomain(LX, i)    = nx / comm.mx;
      meshdomain(XDOWN, i) = meshdomain(LX, i) * meshdomain(XINT, i);
      meshdomain(XUP, i)   = meshdomain(XDOWN, i) + meshdomain(LX, i);
      if (meshdomain(XUP, i) > nx)
        meshdomain(XUP, i) = nx;   
      meshdomain(LY, i)    = ny / comm.my;
      meshdomain(YDOWN, i) = meshdomain(LY, i) * meshdomain(YINT, i);
      meshdomain(YUP, i)   = meshdomain(YDOWN, i) + meshdomain(LY, i);
      if (meshdomain(YUP, i) > ny)
        meshdomain(YUP, i) = ny; 
      meshdomain(SX, i) = meshdomain(LX, i) + 2 * nghost;
      meshdomain(SY, i) = meshdomain(LY, i) + 2 * nghost;
      meshdomain(IMIN, i) = nghost;
      meshdomain(IMAX, i) = meshdomain(LX, i) + nghost;
      meshdomain(JMIN, i) = nghost;
      meshdomain(JMAX, i) = meshdomain(LY, i) + nghost;
    } 

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

  void exchange_T_halo() {
    auto T_left_out_  = this->T_left_out;
    auto T_down_out_  = this->T_down_out;
    auto T_right_out_ = this->T_right_out;
    auto T_up_out_    = this->T_up_out;
    auto T_left_      = this->T_left;
    auto T_down_      = this->T_down;
    auto T_right_     = this->T_right;
    auto T_up_        = this->T_up;
    auto nx_          = this->nx;
    auto ny_          = this->ny;
    auto T_           = this->T;
    auto meshdomain_       = this->meshdomain;
    auto wholeworld_       = this->wholeworld;

    MPI_Request                   mpi_requests_recv[4];
    MPI_Request                   mpi_requests_send[comm.np*4];
    int mar, tag, dest;

    mar          = 0;
    for (int k = 0; k < comm.np; k++) {

      tag = 0 * 4 + meshdomain_(LEFT, k);
      dest = wholeworld_(meshdomain_(LEFT, k));
      Kokkos::deep_copy(T_left_out_, Kokkos::subview(T_, k, 1, Kokkos::ALL));
      comm.isend(
        dest, T_left_out_, tag, &mpi_requests_send[mar]);
      mar++;

      tag = 1 * 4 + meshdomain_(DOWN, k);
      dest = wholeworld_(meshdomain_(DOWN, k));
      Kokkos::deep_copy(T_down_out_, Kokkos::subview(T_, k, Kokkos::ALL, 1));
      comm.isend(
        dest, T_down_out_, tag, &mpi_requests_send[mar]);
      mar++;

      tag = 2 * 4 + meshdomain_(RIGHT, k);
      dest = wholeworld_(meshdomain_(RIGHT, k));
      Kokkos::deep_copy(T_right_out_, Kokkos::subview(T_, k, meshdomain_(LX, k), Kokkos::ALL));
      comm.isend(
        dest, T_right_out_, tag, &mpi_requests_send[mar]);
      mar++;

      tag = 3 * 4 + meshdomain_(UP, k);
      dest = wholeworld_(meshdomain_(UP, k));
      Kokkos::deep_copy(T_up_out_, Kokkos::subview(T_, k, Kokkos::ALL, meshdomain_(LY, k)));
      comm.isend(
        dest, T_up_out_, tag, &mpi_requests_send[mar]);
      mar++;

    }

    mpi_active_requests = mar;
    if (mpi_active_requests > 0) {
      MPI_Waitall(mpi_active_requests, mpi_requests_send, MPI_STATUSES_IGNORE);
    }

    for (int k = 0; k < comm.np; k++) {

      mar          = 0;

      tag = 2 * 4 + meshdomain_(PATCH, k);
      dest = wholeworld_(meshdomain_(LEFT, k));
      comm.irecv(
        dest, T_left_, tag, &mpi_requests_recv[mar]);
      mar++;

      tag = 3 * 4 + meshdomain_(PATCH, k);
      dest = wholeworld_(meshdomain_(DOWN, k));
      comm.irecv(
        dest, T_down_, tag, &mpi_requests_recv[mar]);
      mar++;

      tag = 0 * 4 + meshdomain_(PATCH, k);
      dest = wholeworld_(meshdomain_(RIGHT, k));
      comm.irecv(
        dest, T_right_, tag, &mpi_requests_recv[mar]);
      mar++;

      tag = 1 * 4 + meshdomain_(PATCH, k);
      dest = wholeworld_(meshdomain_(UP, k));
      comm.irecv(
        dest, T_up_, tag, &mpi_requests_recv[mar]);
      mar++;

      mpi_active_requests = mar;
      if (mpi_active_requests > 0) {
        MPI_Waitall(mpi_active_requests, mpi_requests_recv, MPI_STATUSES_IGNORE);
      }

      Kokkos::deep_copy(Kokkos::subview(T_, k, 0, Kokkos::ALL), T_left_);
      Kokkos::deep_copy(Kokkos::subview(T_, k, Kokkos::ALL, 0), T_down_);
      Kokkos::deep_copy(Kokkos::subview(T_, k, meshdomain_(LX, k) + 1, Kokkos::ALL), T_right_);
      Kokkos::deep_copy(Kokkos::subview(T_, k, Kokkos::ALL, meshdomain_(LY, k) + 1), T_up_);

    }

 
  }
 
  void exchange_Ti_halo() {
    auto T_left_out_  = this->T_left_out;
    auto T_down_out_  = this->T_down_out;
    auto T_right_out_ = this->T_right_out;
    auto T_up_out_    = this->T_up_out;
    auto T_left_      = this->T_left;
    auto T_down_      = this->T_down;
    auto T_right_     = this->T_right;
    auto T_up_        = this->T_up;
    auto nx_          = this->nx;
    auto ny_          = this->ny;
    auto Ti_           = this->Ti;
    auto meshdomain_       = this->meshdomain;
    auto wholeworld_       = this->wholeworld;

    MPI_Request                   mpi_requests_recv[4];
    MPI_Request                   mpi_requests_send[comm.np*4];
    int mar, tag, dest;

    mar          = 0;
    for (int k = 0; k < comm.np; k++) {

      tag = 0 * 4 + meshdomain_(LEFT, k);
      dest = wholeworld_(meshdomain_(LEFT, k));
      Kokkos::deep_copy(T_left_out_, Kokkos::subview(Ti_, k, 1, Kokkos::ALL));
      comm.isend(
        dest, T_left_out_, tag, &mpi_requests_send[mar]);
      mar++;

      tag = 1 * 4 + meshdomain_(DOWN, k);
      dest = wholeworld_(meshdomain_(DOWN, k));
      Kokkos::deep_copy(T_down_out_, Kokkos::subview(Ti_, k, Kokkos::ALL, 1));
      comm.isend(
        dest, T_down_out_, tag, &mpi_requests_send[mar]);
      mar++;

      tag = 2 * 4 + meshdomain_(RIGHT, k);
      dest = wholeworld_(meshdomain_(RIGHT, k));
      Kokkos::deep_copy(T_right_out_, Kokkos::subview(Ti_, k, meshdomain_(LX, k), Kokkos::ALL));
      comm.isend(
        dest, T_right_out_, tag, &mpi_requests_send[mar]);
      mar++;

      tag = 3 * 4 + meshdomain_(UP, k);
      dest = wholeworld_(meshdomain_(UP, k));
      Kokkos::deep_copy(T_up_out_, Kokkos::subview(Ti_, k, Kokkos::ALL, meshdomain_(LY, k)));
      comm.isend(
        dest, T_up_out_, tag, &mpi_requests_send[mar]);
      mar++;

    }

    mpi_active_requests = mar;
    if (mpi_active_requests > 0) {
      MPI_Waitall(mpi_active_requests, mpi_requests_send, MPI_STATUSES_IGNORE);
    }

    for (int k = 0; k < comm.np; k++) {

      mar          = 0;

      tag = 2 * 4 + meshdomain_(PATCH, k);
      dest = wholeworld_(meshdomain_(LEFT, k));
      comm.irecv(
        dest, T_left_, tag, &mpi_requests_recv[mar]);
      mar++;

      tag = 3 * 4 + meshdomain_(PATCH, k);
      dest = wholeworld_(meshdomain_(DOWN, k));
      comm.irecv(
        dest, T_down_, tag, &mpi_requests_recv[mar]);
      mar++;

      tag = 0 * 4 + meshdomain_(PATCH, k);
      dest = wholeworld_(meshdomain_(RIGHT, k));
      comm.irecv(
        dest, T_right_, tag, &mpi_requests_recv[mar]);
      mar++;

      tag = 1 * 4 + meshdomain_(PATCH, k);
      dest = wholeworld_(meshdomain_(UP, k));
      comm.irecv(
        dest, T_up_, tag, &mpi_requests_recv[mar]);
      mar++;

      mpi_active_requests = mar;
      if (mpi_active_requests > 0) {
        MPI_Waitall(mpi_active_requests, mpi_requests_recv, MPI_STATUSES_IGNORE);
      }

      Kokkos::deep_copy(Kokkos::subview(Ti_, k, 0, Kokkos::ALL), T_left_);
      Kokkos::deep_copy(Kokkos::subview(Ti_, k, Kokkos::ALL, 0), T_down_);
      Kokkos::deep_copy(Kokkos::subview(Ti_, k, meshdomain_(LX, k) + 1, Kokkos::ALL), T_right_);
      Kokkos::deep_copy(Kokkos::subview(Ti_, k, Kokkos::ALL, meshdomain_(LY, k) + 1), T_up_);

    }

 
  }

  void initialize() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    using policy_3t  = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    auto meshdomain_        = this->meshdomain;
    auto nghost_    = this->nghost;
    auto nx_    = this->nx;
    auto ny_    = this->ny;
    auto T_         = this->T;

    T           = Kokkos::View<double***>("System::T", comm.np, meshdomain(SX, 0), meshdomain(SY, 0));
    Ti          = Kokkos::View<double***>("System::Ti", comm.np, meshdomain(SX, 0), meshdomain(SY, 0));
    dT          = Kokkos::View<double***>("System::dT", comm.np, meshdomain(SX, 0), meshdomain(SY, 0));
    io_recast   = Kokkos::View<double**>("System::io_recast", meshdomain(LX, 0), meshdomain(LY, 0));

    // incoming halos
    T_left      = buffer_t("System::T_left", meshdomain(SY, 0));
    T_right     = buffer_t("System::T_right", meshdomain(SY, 0));
    T_down      = buffer_t("System::T_down", meshdomain(SX, 0));
    T_up        = buffer_t("System::T_up", meshdomain(SX, 0));

    // outgoing halo
    T_left_out  = buffer_t("System::T_left_out", meshdomain(SY, 0));
    T_right_out = buffer_t("System::T_right_out", meshdomain(SY, 0));
    T_down_out  = buffer_t("System::T_down_out", meshdomain(SX, 0));
    T_up_out    = buffer_t("System::T_up_out", meshdomain(SX, 0));

    Kokkos::parallel_for(
            "Init", policy_3t({0, meshdomain_(IMIN, 0), meshdomain_(JMIN, 0) }, {comm.np, meshdomain_(IMAX, 0), meshdomain_(JMAX, 0) }), KOKKOS_LAMBDA(int k, int i, int j){

        T_(k, i, j) = (T1 - T0) * math::exp(-((meshdomain_(XDOWN, k) + i - 0.5 * (nx_ + 2 * nghost_))
                                    * (meshdomain_(XDOWN, k) + i - 0.5 * (nx_ + 2 * nghost_))
                                  + (meshdomain_(YDOWN, k) + j - 0.5 * (ny_ + 2 * nghost_))
                                      * (meshdomain_(YDOWN, k) + j - 0.5 * (ny_ + 2 * nghost_)))
                                / (0.2 * nx_) / (0.2 * nx_));
    });

  
  }

  // Advance physical time
  void evolve(MPI_Comm comm_) {
    Kokkos::Timer timer;
    using policy_t  = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    using policy_3t  = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

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
    auto meshdomain_       = this->meshdomain;
    auto eloc_      = this->eloc;
    auto etot_      = this->etot;
    auto etot0_     = this->etot0;
    // auto imin_      = this->imin;
    // auto imax_      = this->imax;
    // auto jmin_      = this->jmin;
    // auto jmax_      = this->jmax;
    // auto xdown_     = this->xdown;
    // auto xup_       = this->xup;
    // auto ydown_     = this->ydown;
    // auto yup_       = this->yup;

#ifdef OUTPUT_ENABLED
    adios2::ADIOS      adios(comm_);
    // adios2::ADIOS      adios;
    adios2::IO         io = adios.DeclareIO("fieldIO");
    io.SetEngine("HDF5");
    adios2::Dims shape { static_cast<std::size_t>(ny), static_cast<std::size_t>(nx) };
    adios2::Dims start { static_cast<std::size_t>(meshdomain(YDOWN, 0)),
                              static_cast<std::size_t>(meshdomain(XDOWN, 0)) };
    adios2::Dims count { static_cast<std::size_t>(meshdomain(LY, 0)), static_cast<std::size_t>(meshdomain(LX, 0)) };
    auto               io_variable = io.DefineVariable<double>("data", shape, start, count);
    auto               io_mesh = io.DefineVariable<double>("mesh", shape, start, count);
    adios2::Engine adios_engine = io.Open("../Temp.h5", adios2::Mode::Write);
#endif

    printf("My Domain: %i (%i %i) (%i %i) (%i %i) (%i %i)\n", comm.rank, xdown, xup, ydown, yup, nx, ny, lx, ly);
    double time_push = 0.0, time_dump = 0.0, time_bnd = 0.0, time_tot = 0.0;

    Kokkos::parallel_reduce(
      policy_3t({0, meshdomain_(IMIN, 0), meshdomain_(JMIN, 0) }, {comm.np, meshdomain_(IMAX, 0), meshdomain_(JMAX, 0) }),
      KOKKOS_LAMBDA(int k, int i, int j, double& lval) { lval += 0.5 * T_(k, i, j) * T_(k, i, j); },
      eloc_);

    MPI_Reduce(&eloc_, &etot0_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    for (int t = 0; t <= tmax; t++) {
      time_push -= timer.seconds();
      time_tot -= timer.seconds();

      //  Do a Crank-Nicolson stage
      Kokkos::parallel_for(
        "CN1", policy_3t({0, meshdomain_(IMIN, 0), meshdomain_(JMIN, 0) }, {comm.np, meshdomain_(IMAX, 0), meshdomain_(JMAX, 0) }), KOKKOS_LAMBDA(int k, int i, int j) {
          dT_(k, i, j) = 0.5 * vx_ * (T_(k, i + 1, j) - T_(k, i - 1, j));
          dT_(k, i, j) += 0.5 * vy_ * (T_(k, i, j + 1) - T_(k, i, j - 1));
        });

      Kokkos::parallel_for(
        "Euler1", policy_3t({0, meshdomain_(IMIN, 0), meshdomain_(JMIN, 0) }, {comm.np, meshdomain_(IMAX, 0), meshdomain_(JMAX, 0) }), KOKKOS_LAMBDA(int k, int i, int j) {
          Ti_(k, i, j) = T_(k, i, j) + 0.5 * dt_ * dT_(k, i, j);
        });

      time_push += timer.seconds();

      time_bnd -= timer.seconds();

      exchange_Ti_halo();

      time_bnd += timer.seconds();

      time_push -= timer.seconds();

      //  Do final Crank-Nicolson stage
      Kokkos::parallel_for(
        "CN2", policy_3t({0, meshdomain_(IMIN, 0), meshdomain_(JMIN, 0) }, {comm.np, meshdomain_(IMAX, 0), meshdomain_(JMAX, 0) }), KOKKOS_LAMBDA(int k, int i, int j) {
          dT_(k, i, j) = 0.5 * vx_ * (Ti_(k, i + 1, j) - Ti_(k, i - 1, j));
          dT_(k, i, j) += 0.5 * vy_ * (Ti_(k, i, j + 1) - Ti_(k, i, j - 1));
        });

      Kokkos::parallel_for(
        "Euler2", policy_3t({0, meshdomain_(IMIN, 0), meshdomain_(JMIN, 0) }, {comm.np, meshdomain_(IMAX, 0), meshdomain_(JMAX, 0) }), KOKKOS_LAMBDA(int k, int i, int j) {
          T_(k, i, j) = T_(k, i, j) + 0.5 * dt_ * dT_(k, i, j);
        });

      time_push += timer.seconds();

      time_bnd -= timer.seconds();

      exchange_T_halo();

      time_bnd += timer.seconds();

      time_dump -= timer.seconds();

    Kokkos::parallel_reduce(
      policy_3t({0, meshdomain_(IMIN, 0), meshdomain_(JMIN, 0) }, {comm.np, meshdomain_(IMAX, 0), meshdomain_(JMAX, 0) }),
      KOKKOS_LAMBDA(int k, int i, int j, double& lval) { lval += 0.5 * T_(k, i, j) * T_(k, i, j); },
      eloc_);

      MPI_Reduce(&eloc_, &etot_, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      time_dump += timer.seconds();

      if (t % iout == 0 || t == tmax) {
        time_dump -= timer.seconds();

#ifdef OUTPUT_ENABLED

        adios_engine.BeginStep();

    for (int k = 0; k < comm.np; k++) {

        adios2::Dims start { static_cast<std::size_t>(meshdomain_(YDOWN, k)),
                                  static_cast<std::size_t>(meshdomain_(XDOWN, k)) };
        adios2::Dims count { static_cast<std::size_t>(meshdomain_(LY, k)), static_cast<std::size_t>(meshdomain_(LX, k)) };
        io_variable.SetSelection({start, count});
        io_mesh.SetSelection({start, count});
        Kokkos::deep_copy(io_recast_, Kokkos::subview(T_ , k, Kokkos::make_pair(1, meshdomain_(LX, k)+1), Kokkos::make_pair(1, meshdomain_(LY, k)+1)));
        auto outvec = Kokkos::create_mirror_view(io_recast_);
        Kokkos::deep_copy(outvec, io_recast_);
        adios_engine.Put<double>(io_variable, outvec);
        Kokkos::deep_copy(io_recast_, meshdomain_(RANK, k));
        outvec = Kokkos::create_mirror_view(io_recast_);
        Kokkos::deep_copy(outvec, io_recast_);
        adios_engine.Put<double>(io_mesh, outvec);
    
    }
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
