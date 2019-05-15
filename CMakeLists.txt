cmake_minimum_required(VERSION 2.8)
project(GaussianMixtureModels)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "\
    -std=c++11 -O3 -march=native -pthread ${CMAKE_CXX_FLAGS}")
find_package(Eigen3 REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIR}
)


add_executable(${PROJECT_NAME}  main.cpp gaussian_mixture_models.hpp timer.hpp)
target_link_libraries(${PROJECT_NAME}     ${EIGEN3_LIBS})
