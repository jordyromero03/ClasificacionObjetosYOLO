cmake_minimum_required(VERSION 3.20)
project(Practica4_1C_OpenCV_CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Exportar compile_commands.json para clangd/VS Code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimizaciones para i7-11800H
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -DNDEBUG")

# CUDA Toolkit path
set(CUDAToolkit_ROOT /opt/cuda)
set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda)
set(ENV{PATH} "/opt/cuda/bin:$ENV{PATH}")

# OpenCV5 con CUDA
set(OpenCV_DIR /usr/lib/cmake/opencv5)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Verificar soporte CUDA en OpenCV
if(NOT OpenCV_CUDA_VERSION)
    message(WARNING "OpenCV compilado sin CUDA — pipeline GPU no disponible")
else()
    message(STATUS "OpenCV CUDA version: ${OpenCV_CUDA_VERSION}")
endif()

# CUDA Toolkit
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    enable_language(CUDA)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# Ejecutable principal
add_executable(practica4_1c main.cpp)
target_link_libraries(practica4_1c ${OpenCV_LIBS})

# Ejecutable solo benchmark CPU (sin CUDA, para comparación)
add_executable(benchmark_cpu benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu ${OpenCV_LIBS})
