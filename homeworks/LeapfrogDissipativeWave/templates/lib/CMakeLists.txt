cmake_minimum_required(VERSION 3.10)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.23.321.tar.gz"
    SHA1 "5e53cbb0429037ea8e2592bfd92704b8ff3ab492"
    LOCAL
)

project(compgalmat)

message("Project source dir =" "${PROJECT_SOURCE_DIR}")
message("Project binary dir =" "${PROJECT_BINARY_DIR}")

# Check that this is not an in-source build:
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
   message(SEND_ERROR "In-source builds are not allowed.")
endif()

#setup compiler
set(CMAKE_CXX_STANDARD 17)

#enable warnings
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-Wall COMPILER_SUPPORTS_WALL)
CHECK_CXX_COMPILER_FLAG(-Wextra COMPILER_SUPPORTS_WEXTRA)
CHECK_CXX_COMPILER_FLAG(-Wconversion COMPILER_SUPPORTS_WCONVERSION)

add_library(warning_flags INTERFACE)
if(COMPILER_SUPPORTS_WALL)
  target_compile_options(warning_flags INTERFACE -Wall)
endif()
if(COMPILER_SUPPORTS_WEXTRA)
  target_compile_options(warning_flags INTERFACE -Wextra)
endif()
if(COMPILER_SUPPORTS_WCONVERSION)
  target_compile_options(warning_flags INTERFACE -Wconversion)
endif()


# Get Dependencies
# (don't forget to update cmake/Config.cmake.in !
###############################################################################
# Get Eigen
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

# Get LehrFEMpp
hunter_add_package(lehrfempp)
find_package(lehrfempp CONFIG REQUIRED)

# Get Google Test
hunter_add_package(GTest)
find_package(GTest CONFIG REQUIRED)


add_library(compgalmat STATIC compgalmat.cc)
target_link_libraries(compgalmat PUBLIC Eigen3::Eigen LF::lf.base LF::lf.mesh LF::lf.geometry LF::lf.mesh.hybrid2d LF::lf.mesh.utils LF::lf.mesh.test_utils LF::lf.refinement LF::lf.assemble LF::lf.quad LF::lf.io LF::lf.fe LF::lf.uscalfe)
