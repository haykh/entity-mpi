#include <Kokkos_Core.hpp>

#include <iostream>

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> a("a", 10);
    Kokkos::View<bool*>  isbad("isbad", 10);
    Kokkos::parallel_for(
      "fill_a", 10, KOKKOS_LAMBDA(const int i) { a(i) = Kokkos::sqrt(7.0 - i); });
    Kokkos::parallel_for(
      "check_a", 10, KOKKOS_LAMBDA(const int i) { isbad(i) = Kokkos::isfinite(a(i)); });

    std::cout << isbad(0) << " " << isbad(8) << std::endl;
    std::cout << a(0) << " " << a(8) << std::endl;
  }
  Kokkos::finalize();

  return 0;
}
