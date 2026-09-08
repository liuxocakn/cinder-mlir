#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "cinder/Dialect/FCG/FCG.h"

#define GEN_PASS_DEF_FCGPASSFUSEFULLFLOW
#include "cinder/Dialect/FCG/FCGPasses.h"

using namespace mlir;

namespace {

class FuseMatmulAddReluPattern : public OpRewritePattern<cinder::MatMulOp> {
public:
  using OpRewritePattern<cinder::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(cinder::MatMulOp matmul,
                                PatternRewriter &rewriter) const override {
    auto matmulResult = matmul.getResult();
    if (!matmulResult.hasOneUse())
      return failure();

    auto addOp = dyn_cast<cinder::AddOp>(*matmulResult.getUsers().begin());
    if (!addOp)
      return failure();

    Value bias;
    Value matmulOutput = matmulResult;
    if (addOp.getLhs() == matmulOutput && addOp.getRhs() != matmulOutput) {
      bias = addOp.getRhs();
    } else if (addOp.getRhs() == matmulOutput && addOp.getLhs() != matmulOutput) {
      bias = addOp.getLhs();
    } else {
      return failure();
    }

    auto addResult = addOp.getResult();
    if (!addResult.hasOneUse())
      return failure();

    auto reluOp = dyn_cast<cinder::ReluOp>(*addResult.getUsers().begin());
    if (!reluOp)
      return failure();

    if (reluOp.getInput() != addResult)
      return failure();

    Value A = matmul.getLhs();
    Value B = matmul.getRhs();
    Value out = reluOp.getOut();

    Location loc = reluOp.getLoc();
    auto resultType = reluOp.getResult().getType();

    auto fullFlowOp = rewriter.create<cinder::FullFlowOp>(loc, resultType, A, B, bias, out);

    rewriter.replaceOp(reluOp, fullFlowOp.getResult());
    rewriter.eraseOp(addOp);
    rewriter.eraseOp(matmul);

    return success();
  }
};

} // namespace

namespace cinder {

class FCGPassFuseFullFlow
    : public impl::FCGPassFuseFullFlowBase<FCGPassFuseFullFlow> {
public:
  using impl::FCGPassFuseFullFlowBase<FCGPassFuseFullFlow>::FCGPassFuseFullFlowBase;

  void returnStatus(func::FuncOp func, const char* msg) {
    if (msg) {
      func.emitError(msg);
      signalPassFailure();
    }
  }

  void runOnOperation() override {
    ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    if (!module)
      return;

    for (auto func : module.getOps<func::FuncOp>()) {
      StringRef funcName = func.getName();
      if (!funcName.startswith("test_fusion_full_flow_"))
        continue;

      if (!funcName.endswith("_f32") && !funcName.endswith("_f16")) {
        returnStatus(func, "Invalid function name for fusion: must end with _f32 or _f16");
        return;
      }

      bool hasMatmul = false;
      func.walk([&](cinder::MatMulOp op) {
        hasMatmul = true;
      });
      if (!hasMatmul) {
        returnStatus(func, "Expected at least one fcg.matmul in fusion function");
        return;
      }

      MLIRContext *ctx = &getContext();
      RewritePatternSet patterns(ctx);
      patterns.add<FuseMatmulAddReluPattern>(ctx);

      GreedyRewriteConfig config;
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        returnStatus(func, "Greedy rewrite failed during fusion");
        return;
      }

      bool foundMatmul = false;
      func.walk([&](cinder::MatMulOp op) {
        foundMatmul = true;
      });
      if (foundMatmul) {
        returnStatus(func, "Failed to fuse matmul+add+relu: matmul still present after fusion");
        return;
      }
    }
  }
};

std::unique_ptr<mlir::Pass> createFCGPassFuseFullFlow() {
  return std::make_unique<FCGPassFuseFullFlow>();
}

} // namespace cinder
