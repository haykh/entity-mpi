set(EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../extern)
set(ENTITY_MPI ${PROJECT_NAME}.xc)

find_package(ADIOS2 REQUIRED)

include_directories(${EXTERN_DIR})
link_libraries(Kokkos::kokkos)

add_executable(${ENTITY_MPI} main.cpp)