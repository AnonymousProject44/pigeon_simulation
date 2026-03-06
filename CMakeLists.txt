cmake_minimum_required(VERSION 3.28)

project(simulator)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${CMAKE_SOURCE_DIR}/install")
endif()

find_package(Eigen3)
find_package(Boost)

add_executable(simulator)
target_sources(simulator
  PRIVATE
    "src/main.cpp"
)

target_include_directories(simulator
  PRIVATE
    "include"
    "include/mrs_multirotor_simulator/uav_system/ode"
)

target_link_libraries(simulator
  PRIVATE
    Eigen3::Eigen
    Boost::headers
)

install(
  TARGETS simulator
)