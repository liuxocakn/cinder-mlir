add_mlir_tool(cinder-opt cinder-opt.cpp)

target_compile_options(cinder-opt PRIVATE -O0 -g3)

target_link_libraries(cinder-opt PRIVATE
    FCGIR
    FCGTransforms
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRTransforms
    MLIROptLib
    MLIRArithDialect
    LLVMSupport

    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
)

install(TARGETS cinder-opt RUNTIME DESTINATION ${CINDER_INSTALL_PATH})
