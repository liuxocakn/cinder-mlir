#ifndef __FCGOPS_TD__
#define __FCGOPS_TD__

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "cinder/Dialect/FCG/FCGDialect.td"

class FCG_Op<string mnemonic, list<Trait> traits = []>
    : Op<FCGDialect, mnemonic, traits>;

class FCG_TensorOp<string mnemonic, list<Trait> traits = []>
    : FCG_Op<mnemonic, !listconcat(traits, [Pure])>;

def MatMulOp : FCG_TensorOp<"matmul"> {
  let arguments = (ins
    TensorOf<[F32, F16]>:$lhs,
    TensorOf<[F32, F16]>:$rhs,
    TensorOf<[F32, F16]>:$out
  );
  let results = (outs TensorOf<[F32, F16]>:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $out attr-dict `:` type($lhs) `,` type($rhs) `,` type($out) `->` type($result)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static StringRef getKernelNameF32() { return "gpu_matmul_f32"; }
    static StringRef getKernelNameF16() { return "gpu_matmul_f16"; }
  }];
}

def AddOp : FCG_TensorOp<"add", [Commutative]> {
  let arguments = (ins
    TensorOf<[F32, F16]>:$lhs,
    TensorOf<[F32, F16]>:$rhs,
    TensorOf<[F32, F16]>:$out
  );
  let results = (outs TensorOf<[F32, F16]>:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $out attr-dict `:` type($lhs) `,` type($rhs) `,` type($out) `->` type($result)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static StringRef getKernelNameF32() { return "gpu_add_f32"; }
    static StringRef getKernelNameF16() { return "gpu_add_f16"; }
  }];
}

def ReluOp : FCG_TensorOp<"relu"> {
  let arguments = (ins
    TensorOf<[F32, F16]>:$input,
    TensorOf<[F32, F16]>:$out
  );
  let results = (outs TensorOf<[F32, F16]>:$result);
  let assemblyFormat = "$input `,` $out attr-dict `:` type($input) `,` type($out) `->` type($result)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static StringRef getKernelNameF32() { return "gpu_relu_f32"; }
    static StringRef getKernelNameF16() { return "gpu_relu_f16"; }
  }];
}

def FullFlowOp: FCG_TensorOp<"fullflow"> {
  let summary = "fused matmul + add + relu";
  let description = [{
    Performs `relu(matmul(lhs, rhs) + bias)` in a single fused operation,
    reducing intermediate memory traffic.
  }];
  let arguments = (ins
    TensorOf<[F32, F16]>:$lhs,
    TensorOf<[F32, F16]>:$rhs,
    TensorOf<[F32, F16]>:$bias,
    TensorOf<[F32, F16]>:$out
  );
  let results = (outs TensorOf<[F32, F16]>:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $bias `,` $out attr-dict `:` type($lhs) `,` type($rhs) `,` type($bias) `,` type($out) `->` type($result)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static StringRef getKernelNameF32() { return "gpu_fullflow_f32"; }
    static StringRef getKernelNameF16() { return "gpu_fullflow_f16"; }
  }];
}

#endif
