cmake_minimum_required(VERSION 3.2 FATAL_ERROR)


project(hfcpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get dependencies
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include(FindOpenMP)
add_definitions(-DOMP_NUM_THREADS=8)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/arch)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# compiler flags
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    message("Detected GNU-Compiler.")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_BUILD_TYPE Release)
        set(COVERAGE_COMPILE_FLAGS "-Wall -Weffc++ -Wextra -Werror -pedantic -lpthread -O3")
        set(COVERAGE_LINK_FLAGS    "")
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_BUILD_TYPE Debug)
        set(COVERAGE_COMPILE_FLAGS "-Wall -Weffc++ -Wextra -Werror -pedantic -lpthread -g -O0 -D_GLIBCXX_DEBUG")
        set(COVERAGE_LINK_FLAGS    "")
    else()
        set(CMAKE_BUILD_TYPE Release)
        set(COVERAGE_COMPILE_FLAGS "-Wall -Weffc++ -Wextra -Werror -pedantic -lpthread -O3")
        set(COVERAGE_LINK_FLAGS    "")
    endif()

elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    message("Detected Clang-Compiler.")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(CMAKE_BUILD_TYPE Release)
        set(COVERAGE_COMPILE_FLAGS "-Wall -Wextra -Werror -pedantic -pthread -O3")
        set(COVERAGE_LINK_FLAGS    "")
    elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_BUILD_TYPE Debug)
        set(COVERAGE_COMPILE_FLAGS "-Wall -Wextra -Werror -pedantic -pthread -g -O0 -fsanitize=address")
        set(COVERAGE_LINK_FLAGS    "")
    else()
        set(CMAKE_BUILD_TYPE Release)
        set(COVERAGE_COMPILE_FLAGS "-Wall -Wextra -Werror -pedantic -pthread -O3")
        set(COVERAGE_LINK_FLAGS    "")
    endif()
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${COVERAGE_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${COVERAGE_LINK_FLAGS}")


add_executable(hfcpp)
target_sources(hfcpp
    PRIVATE
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_SOURCE_DIR}/src/scf.cpp"
    )

target_link_libraries(hfcpp PUBLIC Eigen3::Eigen)
