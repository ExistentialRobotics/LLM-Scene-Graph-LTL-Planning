find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(Boost REQUIRED COMPONENTS program_options graph)
add_subdirectory(3rd_party/nanoflann)
find_package(yaml-cpp REQUIRED)
find_package(absl REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(spot REQUIRED libspot)

file(GLOB SRC_FILES src/erl_common/*.cpp)
file(GLOB SRC_FILES ${SRC_FILES} src/erl_env/*.cpp)
file(GLOB SRC_FILES ${SRC_FILES} src/erl_search_planning/*.cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color -fdiagnostics-show-template-tree")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops")
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DNDEBUG)
else ()
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif ()

add_library(amra_ltl_scene_graph_lib ${SRC_FILES})
target_include_directories(amra_ltl_scene_graph_lib PUBLIC include)
target_include_directories(amra_ltl_scene_graph_lib PUBLIC SYSTEM ${OpenCV_INCLUDE_DIRS})
target_include_directories(amra_ltl_scene_graph_lib PUBLIC SYSTEM ${spot_INCLUDE_DIRS})
target_link_libraries(amra_ltl_scene_graph_lib
        PUBLIC ${OpenCV_LIBRARIES}
        PUBLIC Eigen3::Eigen
        PUBLIC pthread OpenMP::OpenMP_CXX
        PUBLIC Boost::boost Boost::program_options Boost::graph
        PUBLIC nanoflann::nanoflann
        PUBLIC yaml-cpp
        PUBLIC absl::flat_hash_map absl::flat_hash_set
        PUBLIC ${spot_LIBRARIES}
)

add_executable(amra_ltl_scene_graph src/apps/amra_ltl_scene_graph.cpp)
target_link_libraries(amra_ltl_scene_graph PRIVATE amra_ltl_scene_graph_lib)

option(BUILD_TEST "Build test" OFF)
if (BUILD_TEST)
    add_subdirectory(test)
endif ()
