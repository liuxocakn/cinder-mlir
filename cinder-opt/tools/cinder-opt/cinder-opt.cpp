#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "cinder/Dialect/FCG/FCG.h"
#include "cinder/Dialect/FCG/FCGPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<cinder::FCGDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  cinder::registerPasses();

  mlir::registerPassManagerCLOptions();
  mlir::registerAsmPrinterCLOptions();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Cinder MLIR optimizer with FCG", registry)
  );
}
