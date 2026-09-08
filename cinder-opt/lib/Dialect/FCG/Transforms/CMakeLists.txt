add_mlir_library(FCGTransforms

  FCGPassFuseFullFlow.cpp
  FCGPassBaseline.cpp
  FCGPassVectorize.cpp
  FCGPassLinalg.cpp
  FCGPassGPU.cpp
  FCGPassNVVMGPU.cpp
  FCGPassNVVMCPU.cpp

  DEPENDS
  MLIRFCGIncGen
  MLIRFCGPassesIncGen
  MLIRArithDialect
  MLIRVectorDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRGPUDialect
  MLIRNVVMDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRBufferizationDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRVectorDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRGPUDialect
  MLIRNVVMDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  FCGIR

  DISABLE_INSTALL
)

target_compile_options(obj.FCGTransforms PRIVATE -Wno-overloaded-virtual)
target_compile_options(obj.FCGTransforms PRIVATE -O0 -g3)
