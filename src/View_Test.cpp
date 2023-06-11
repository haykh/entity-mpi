#include <Kokkos_Core.hpp>
#include <iostream>

auto main(int argc, char** argv) -> int {

  Kokkos::initialize(argc, argv);
  {
    Kokkos::Timer         timer;
    double                time_push;

    int                   number_of_doubles   = 1000000;
    int                   number_dims         = 32;
    int                   sample_size         = 100000;
    Kokkos::View<double*> view1("view1", number_of_doubles);
    Kokkos::View<double**> view2("view2", number_of_doubles, number_dims);
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

    time_push           = 0.0;
    for (int t = 0; t <= sample_size; t++) {
      
      time_push -= timer.seconds();

      Kokkos::parallel_for(
          number_of_doubles, KOKKOS_LAMBDA(int i) { view1(i) = 0.0; });
      Kokkos::parallel_for(
          number_of_doubles, KOKKOS_LAMBDA(int i) { view1(i) = 1.0*i; });

      time_push += timer.seconds();

    }

    auto outvec1 = Kokkos::create_mirror_view(view1);
    Kokkos::deep_copy(outvec1, view1);
    printf(" Test %lf \n", outvec1(100));
    printf(" Timing %lf \n", time_push/sample_size);

    time_push           = 0.0;
    for (int t = 0; t <= sample_size; t++) {
      
      time_push -= timer.seconds();

      Kokkos::parallel_for(
            "View2", policy_t({0, 0}, {number_of_doubles, number_dims}), KOKKOS_LAMBDA(int i, int j) {
              view2(i,j) = 0.0;
            });

      Kokkos::parallel_for(
            "View2", policy_t({0, 0}, {number_of_doubles, number_dims}), KOKKOS_LAMBDA(int i, int j) {
              view2(i,j) = 1.0*i;
            });

      time_push += timer.seconds();

    }

    auto outvec2 = Kokkos::create_mirror_view(view2);
    Kokkos::deep_copy(outvec2, view2);
    printf(" Test %lf \n", outvec2(100,0));
    printf(" Timing %lf \n", time_push/sample_size);

    time_push           = 0.0;
    for (int t = 0; t <= sample_size; t++) {
      
      time_push -= timer.seconds();

      Kokkos::parallel_for(
          number_of_doubles*number_dims, KOKKOS_LAMBDA(int i) { int ii = Kokkos::floor(i/number_of_doubles); int jj = i%number_of_doubles; view2(jj, ii) = 0.0; });
      Kokkos::parallel_for(
          number_of_doubles*number_dims, KOKKOS_LAMBDA(int i) { int ii = Kokkos::floor(i/number_of_doubles); int jj = i%number_of_doubles; view2(jj, ii) = jj; });

      time_push += timer.seconds();

    }

    auto outvec3 = Kokkos::create_mirror_view(view2);
    Kokkos::deep_copy(outvec3, view2);
    printf(" Test %lf \n", outvec3(100,0));
    printf(" Timing %lf \n", time_push/sample_size);

  }
  Kokkos::finalize();

  return 0;
}
