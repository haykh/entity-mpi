set(EXTERN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../extern)
set(ENTITY_MPI ${PROJECT_NAME}.xc)

include_directories(${EXTERN_DIR})
link_libraries(Kokkos::kokkos)

if(${output} STREQUAL "ON")
  link_libraries(adios2::cxx11)
endif()

add_executable(${ENTITY_MPI} main.cpp)