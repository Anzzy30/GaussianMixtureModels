cmake_minimum_required(VERSION 2.8)
project(GaussianMixtureModels CXX C)
set(CMAKE_CXX_FLAGS "\
    -std=c++11 -O3 -march=native -pthread ${CMAKE_CXX_FLAGS}")

find_package(Eigen3 REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(
        ${PROJECT_NAME} SHARED
        include/gaussian_mixture_models.hpp
        include/gaussian_mixture_models.cpp
        include/timer.hpp
)

target_link_libraries(
    ${PROJECT_NAME}
    ${EIGEN3_LIBS}
)

add_executable(GaussianTest main.cpp )
target_link_libraries(GaussianTest ${PROJECT_NAME})

