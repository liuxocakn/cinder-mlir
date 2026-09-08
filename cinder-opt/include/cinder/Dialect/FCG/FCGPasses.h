#ifndef __FCGPASSES_H__
#define __FCGPASSES_H__

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace cinder {

#define GEN_PASS_DECL
#include "cinder/Dialect/FCG/FCGPasses.h.inc"

std::unique_ptr<mlir::Pass> createFCGPassFuseFullFlow();
std::unique_ptr<mlir::Pass> createFCGPassBaseline();
std::unique_ptr<mlir::Pass> createFCGPassVectorize();
std::unique_ptr<mlir::Pass> createFCGPassLinalg();
std::unique_ptr<mlir::Pass> createFCGPassGPU();
std::unique_ptr<mlir::Pass> createFCGPassNVVMGPU();
std::unique_ptr<mlir::Pass> createFCGPassNVVMCPU();

#define GEN_PASS_REGISTRATION
#include "cinder/Dialect/FCG/FCGPasses.h.inc"

}

#endif // __FCGPASSES_H__
