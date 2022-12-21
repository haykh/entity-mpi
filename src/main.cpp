#include <Kokkos_Core.hpp>

#include <iostream>

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {}
  Kokkos::finalize();

  return 0;
}
