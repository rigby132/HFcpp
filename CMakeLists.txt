cmake_minimum_required(VERSION 3.2 FATAL_ERROR)


project(hf)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# Get dependencies
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include(FindOpenMP)
add_definitions(-DOMP_NUM_THREADS=4)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/arch)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# compiler flags
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Werror -pedantic -lpthread -Ofast")
set(GCC_COVERAGE_LINK_FLAGS    "")

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")


add_executable(hf)
target_sources(hf
    PRIVATE
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_SOURCE_DIR}/src/scf.cpp"
    "${CMAKE_SOURCE_DIR}/src/basis.cpp"
    )

target_link_libraries(hf Eigen3::Eigen)
