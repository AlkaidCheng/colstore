# CMake build for the colstore native extension.
#
# Builds a single Python extension module (``colstore._gather``) from a
# Cython binding plus a C++ implementation. OpenMP is linked in when the
# toolchain supports it (the kernel still works without OpenMP, just
# single-threaded inside the inner loop).
#
# This file is driven by scikit-build-core via pyproject.toml. To build
# manually without the Python packaging:
#     cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(which python)
#     cmake --build build

cmake_minimum_required(VERSION 3.18)
project(colstore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Dependencies --------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(OpenMP)

# --- Cython preprocess ---------------------------------------------------
# Invoke Cython at build time to translate _gather.pyx into _gather.cpp.
# Done with a custom command rather than a CMake module so the user only
# needs Cython installed via pip, not a CMake find module.
set(CYTHON_PYX "${CMAKE_CURRENT_SOURCE_DIR}/src/cython/_gather.pyx")
set(CYTHON_CPP "${CMAKE_CURRENT_BINARY_DIR}/_gather.cpp")

add_custom_command(
  OUTPUT ${CYTHON_CPP}
  COMMAND ${Python_EXECUTABLE} -m cython
          --cplus -3
          -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
          -o "${CYTHON_CPP}"
          "${CYTHON_PYX}"
  DEPENDS ${CYTHON_PYX}
          "${CMAKE_CURRENT_SOURCE_DIR}/include/colstore/gather.hpp"
  COMMENT "Cythonizing _gather.pyx -> _gather.cpp"
  VERBATIM)

# --- Extension module ----------------------------------------------------
Python_add_library(_gather MODULE WITH_SOABI
  ${CYTHON_CPP}
  src/cpp/gather.cpp
)

target_include_directories(_gather PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Python_NumPy_INCLUDE_DIRS}
)

# Portable optimization flags. We deliberately do not pass -march=native so
# wheels stay portable across CPUs; users who build from source for a
# specific machine can override CMAKE_CXX_FLAGS_RELEASE.
target_compile_options(_gather PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -ffast-math -funroll-loops>
  $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast>
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_gather PRIVATE OpenMP::OpenMP_CXX)
endif()

# Quiet a NumPy 2.x deprecation that fires harmlessly during Cython compile
# of array-API users that don't opt into the new API yet.
target_compile_definitions(_gather PRIVATE
  NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
)

# --- Install -------------------------------------------------------------
# scikit-build-core copies this into the wheel under wheel.packages.
install(TARGETS _gather DESTINATION colstore)
