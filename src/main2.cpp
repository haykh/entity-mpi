#include <Kokkos_Macros.hpp>

#include <iostream>
#include <sstream>

#define USE_MPI
#if defined(USE_MPI)
#  include <mpi.h>
#endif

#include <Kokkos_Core.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

int main(int argc, char** argv) {
  // std::ostringstream msg;

  (void)argc;
  (void)argv;

#if defined(USE_MPI)

  MPI_Init(&argc, &argv);

  // int mpi_rank = 0;

  // MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // msg << "MPI rank(" << mpi_rank << ") ";

#endif

  Kokkos::initialize(argc, argv);
  {
    // msg << "{" << std::endl;

    // if (Kokkos::hwloc::available()) {
    //   msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
    //       << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
    //       << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
    // }

    // Kokkos::print_configuration(msg);

    // msg << "}" << std::endl;

    // std::cout << msg.str();

    int                   source_rank       = 0;
    int                   destination_rank  = 1;
    int                   number_of_doubles = 12;
    int                   tag               = 42;
    Kokkos::View<double*> buffer("buffer", number_of_doubles);
    int                   my_rank;
    MPI_Comm              comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Status mpi_requests_recv[1];
    if (my_rank == source_rank) {
      MPI_Send(buffer.data(), int(buffer.size()), MPI_DOUBLE, destination_rank, tag, comm);
    } else if (my_rank == source_rank) {
      MPI_Recv(buffer.data(), int(buffer.size()), MPI_DOUBLE, source_rank, tag, comm, &mpi_requests_recv[0]);
    }
  }
  Kokkos::finalize();

#if defined(USE_MPI)

  MPI_Finalize();

#endif

  return 0;
}
