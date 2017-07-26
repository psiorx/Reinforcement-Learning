project(rl)

cmake_minimum_required(VERSION 2.8)
find_package(Eigen3 REQUIRED)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB RL_SRC "src/*.cpp")

add_executable(tictactoe 
  ${RL_SRC}
)
