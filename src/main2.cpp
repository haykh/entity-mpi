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
  std::ostringstream msg;

  (void)argc;
  (void)argv;

#if defined(USE_MPI)

  MPI_Init(&argc, &argv);

  int mpi_rank = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // msg << "MPI rank(" << mpi_rank << ") ";

#endif

printf("Hello world from processor rank %d \n", mpi_rank);


  // Kokkos::initialize(argc, argv);
  // {
  //   msg << "{" << std::endl;

  //   if (Kokkos::hwloc::available()) {
  //     msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count() << "] x CORE["
  //         << Kokkos::hwloc::get_available_cores_per_numa() << "] x HT["
  //         << Kokkos::hwloc::get_available_threads_per_core() << "] )" << std::endl;
  //   }

  //   Kokkos::print_configuration(msg);

  //   msg << "}" << std::endl;

  //   std::cout << msg.str();

  //   int                   source_rank       = 0;
  //   int                   destination_rank  = 1;
  //   int                   number_of_doubles = 3;
  //   int                   tag               = 42;
  //   Kokkos::View<double*> buffer("buffer", number_of_doubles);
  //   int                   my_rank;
  //   MPI_Comm              comm = MPI_COMM_WORLD;
  //   MPI_Comm_rank(comm, &my_rank);

  //   Kokkos::parallel_for(number_of_doubles, KOKKOS_LAMBDA(int i) {
  //       buffer(i) = i;
  //     });

  //   MPI_Status mpi_requests_recv[1];
  //   if (my_rank == source_rank) {
  //     auto outvec = Kokkos::create_mirror_view(buffer);
  //     Kokkos::deep_copy(outvec, buffer);
  //     MPI_Send(outvec.data(), int(outvec.size()), MPI_DOUBLE, destination_rank, tag, comm);
  //     std::cout << outvec(0) << outvec(1) << outvec(2) << std::endl;
  //   } else if (my_rank == destination_rank) {
  //     auto outvec = Kokkos::create_mirror_view(buffer);
  //     // Kokkos::deep_copy(outvec, buffer);
  //     MPI_Recv(outvec.data(),
  //              int(outvec.size()),
  //              MPI_DOUBLE,
  //              source_rank,
  //              tag,
  //              comm,
  //              &mpi_requests_recv[0]);
  //     std::cout << outvec(0) << outvec(1) << outvec(2) << std::endl;
  //   }
  // }
  // Kokkos::finalize();

#if defined(USE_MPI)

  MPI_Finalize();

#endif

  return 0;
}
