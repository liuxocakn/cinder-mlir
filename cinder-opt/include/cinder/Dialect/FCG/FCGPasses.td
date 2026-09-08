#ifndef __FCG_FCG_PASSES_TD__
#define __FCG_FCG_PASSES_TD__

include "mlir/Pass/PassBase.td"

def FCGPassFuseFullFlow : Pass<"fcg-pass-fuse-full-flow"> {
  let summary = "Fuse fcg.matmul, fcg.add, fcg.relu into fcg.fullflow";
  let constructor = "cinder::createFCGPassFuseFullFlow()";
}

def FCGPassBaseline : Pass<"fcg-pass-baseline"> {
  let summary = "Lower FCG ops to scf.for + memref scalar loops";
  let constructor = "cinder::createFCGPassBaseline()";
}

def FCGPassVectorize : Pass<"fcg-pass-vectorize"> {
  let summary = "Lower FCG ops to vectorized scf.for + memref loops";
  let constructor = "cinder::createFCGPassVectorize()";
}

def FCGPassLinalg : Pass<"fcg-pass-linalg"> {
  let summary = "Lower FCG ops to linalg.generic / linalg.matmul";
  let constructor = "cinder::createFCGPassLinalg()";
}

def FCGPassGPU : Pass<"fcg-pass-gpu"> {
  let summary = "Lower FCG ops to gpu.launch kernels";
  let constructor = "cinder::createFCGPassGPU()";
}

def FCGPassNVVMGPU : Pass<"fcg-pass-nvvm-gpu", "mlir::ModuleOp"> {
  let summary = "Lower FCG ops directly to nvvm kernels";
  let constructor = "cinder::createFCGPassNVVMGPU()";
}

def FCGPassNVVMCPU : Pass<"fcg-pass-nvvm-cpu", "mlir::ModuleOp"> {
  let summary = "Lower FCG ops to CUDA driver API launch wrappers";
  let constructor = "cinder::createFCGPassNVVMCPU()";
  let options = [
    Option<"kernelPath", "kernel-path", "std::string", /*default=*/"\"libFCGPassNVVM.so.cubin\"", "Path to embedded kernel data file">
  ];
}

#endif
