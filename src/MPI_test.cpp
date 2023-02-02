#if defined(MPI_ENABLED)
#  include <mpi.h>
#endif

#include <Kokkos_Core.hpp>

#include <iostream>

auto main(int argc, char** argv) -> int {
#if defined(MPI_ENABLED)
  MPI_Init(&argc, &argv);
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

  Kokkos::initialize(argc, argv);
  {
    Kokkos::Timer         timer;
    double                time_push           = 0.0;
    double                time_push_glob           = 0.0;

    int                   source_rank         = 0;
    int                   destination_rank    = 1;
    int                   number_of_doubles   = 10;
    int                   tag                 = 42;
    Kokkos::View<double*> sbuffer("sbuffer", number_of_doubles);
    Kokkos::View<double*> rbuffer("rbuffer", number_of_doubles);

    MPI_Status mpi_requests_send[1];
    // MPI_Request mpi_requests_recv[1];

    for (int t = 0; t <= 10000; t++) {
    int                   mpi_active_requests = 0;

      Kokkos::parallel_for(
        number_of_doubles, KOKKOS_LAMBDA(int i) { sbuffer(i) = 1.0*i; });

      Kokkos::parallel_for(
        number_of_doubles, KOKKOS_LAMBDA(int i) { rbuffer(i) = 0.0; });

    time_push -= timer.seconds();

    // if (mpi_rank == source_rank) {
    //   // printf("Hello world from processor rank %d \n", mpi_rank);
    //   MPI_Isend(sbuffer.data(), sbuffer.size(), MPI_DOUBLE, destination_rank, tag, MPI_COMM_WORLD, &mpi_requests_send[0]);
    //   MPI_Irecv(rbuffer.data(), rbuffer.size(), MPI_DOUBLE, destination_rank, tag, MPI_COMM_WORLD, &mpi_requests_recv[0]); 
    //   mpi_active_requests += 1;
    // } else if (mpi_rank == destination_rank) {
    //   // printf("Hello world from processor rank %d \n", mpi_rank);
    //   MPI_Isend(sbuffer.data(), sbuffer.size(), MPI_DOUBLE, source_rank, tag, MPI_COMM_WORLD, &mpi_requests_send[0]);
    //   MPI_Irecv(rbuffer.data(), rbuffer.size(), MPI_DOUBLE, source_rank, tag, MPI_COMM_WORLD, &mpi_requests_recv[0]); 
    //   mpi_active_requests += 1;
    // }

    // MPI_Waitall(mpi_active_requests, mpi_requests_send, MPI_STATUSES_IGNORE);
    // MPI_Waitall(mpi_active_requests, mpi_requests_recv, MPI_STATUSES_IGNORE);

    if (mpi_rank == source_rank) {
      // printf("Hello world from processor rank %d \n", mpi_rank);
        MPI_Sendrecv(sbuffer.data(), sbuffer.size(), MPI_DOUBLE, destination_rank, tag, rbuffer.data(), rbuffer.size(), MPI_DOUBLE, destination_rank, tag, MPI_COMM_WORLD, &mpi_requests_send[0]);
    } else if (mpi_rank == destination_rank) {
      // printf("Hello world from processor rank %d \n", mpi_rank);
        MPI_Sendrecv(sbuffer.data(), sbuffer.size(), MPI_DOUBLE, source_rank, tag, rbuffer.data(), rbuffer.size(), MPI_DOUBLE, source_rank, tag, MPI_COMM_WORLD, &mpi_requests_send[0]);
    }

    time_push += timer.seconds();

      // auto outvec = Kokkos::create_mirror_view(rbuffer);
      // Kokkos::deep_copy(outvec, rbuffer);
      // printf(" Test %lf \n", outvec(1));

    }

    MPI_Reduce(&time_push, &time_push_glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


    if (mpi_rank == 0) {
      printf(" Timing %lf \n", time_push_glob/(2.0*10000.0));
    }

  }
  Kokkos::finalize();

#if defined(MPI_ENABLED)

  MPI_Finalize();

#endif

  return 0;
}
