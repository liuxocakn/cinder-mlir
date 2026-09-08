cmake_minimum_required(VERSION 3.13.4)
cmake_policy(SET CMP0116 NEW)

project(cinder-mlir VERSION 0.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CINDER_BINDING_PATH ${CMAKE_SOURCE_DIR}/binding)
set(CINDER_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/test/lib)
set(CINDER_INSTALL_PATH ${CMAKE_SOURCE_DIR}/test/bin)

set(CINDER_CUDA_LINKAGE
	"-L/usr/local/cuda/lib64"
	"-L/usr/lib/wsl/lib"
	"-L${CMAKE_SOURCE_DIR}/llvm-mlir-bin/lib"
	"-lmlir_cuda_runtime"
	"-lcuda"
	"-lcudart"
)
set(CINDER_CUDA_INCLUDE
    "/usr/local/cuda/include"
)
set(CINDER_CUDA_SMLEVEL
	"sm_86"
)

get_filename_component(MLIR_TOOL_DIR "${MLIR_TOOL_DIR}" ABSOLUTE)

find_program(MLIR_OPT       mlir-opt       HINTS ${MLIR_TOOL_DIR}/bin)
find_program(MLIR_TRANSLATE mlir-translate HINTS ${MLIR_TOOL_DIR}/bin)
find_program(LLC            llc            HINTS ${MLIR_TOOL_DIR}/bin)
find_program(CLANG          clang)
find_program(GCC            gcc)
find_program(PTXAS          ptxas)

foreach(tool MLIR_OPT MLIR_TRANSLATE CLANG LLC PTXAS GCC)
    if(NOT ${tool})
        message(WARNING "Tool ${tool} not found in PATH. Build may fail.")
    endif()
endforeach()

add_subdirectory(cinder-opt)
add_subdirectory(binding)
add_subdirectory(test/c)
