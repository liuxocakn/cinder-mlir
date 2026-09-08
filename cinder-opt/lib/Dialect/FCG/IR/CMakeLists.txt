add_mlir_library(FCGIR

  FCGDialect.cpp
  FCGOps.cpp

  DEPENDS
  MLIRFCGIncGen
  MLIRIR

  DISABLE_INSTALL
)

target_compile_options(obj.FCGIR PRIVATE -O0 -g3)
