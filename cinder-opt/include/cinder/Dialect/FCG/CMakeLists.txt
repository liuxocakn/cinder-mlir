add_mlir_dialect(FCG fcg)

set(LLVM_TARGET_DEFINITIONS FCGPasses.td)
mlir_tablegen(FCGPasses.h.inc -gen-pass-decls -dialect=fcg)
add_public_tablegen_target(MLIRFCGPassesIncGen)
