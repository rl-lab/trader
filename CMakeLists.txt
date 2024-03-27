cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)

set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -I .. -Wall -mcmodel=medium -march=native -Wextra -Wno-register -fPIC -Wfatal-errors -O3")
  set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")


project(trade_env)


find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)


include_directories(
    ${PYTHON_INCLUDE_DIRS}
)


pybind11_add_module(trade_env src/envapi.cpp)
target_include_directories(trade_env PUBLIC 
    ${PYTHON_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/src/ )
target_link_libraries(trade_env PUBLIC)


