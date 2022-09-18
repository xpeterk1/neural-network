project(neural_network)

find_package(OpenMP)

cmake_minimum_required(VERSION 2.8.4)
set(CMAKE_CXX_STANDARD 17)

add_executable(neural_network src/main.cpp src/main.h src/matrix/MatrixOperations.cpp src/matrix/MatrixOperations.h)

target_link_libraries(neural_network PUBLIC OpenMP::OpenMP_CXX)
