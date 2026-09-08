#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "cinder/Dialect/FCG/FCG.h"

#define GEN_PASS_DEF_FCGPASSBASELINE
#include "cinder/Dialect/FCG/FCGPasses.h"

using namespace mlir;
using namespace cinder;

namespace {

static Type convertTensorToMemRef(Type type) {
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  }
  return type;
}

struct MatMulOpConversion : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value out = adaptor.getOut();

    auto outType = out.getType().cast<MemRefType>();
    if (outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "only 2D tensors supported");

    Location loc = op.getLoc();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);
    Value K = rewriter.create<memref::DimOp>(loc, lhs, 1);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, M, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value i = loopI.getInductionVar();

    auto loopJ = rewriter.create<scf::ForOp>(loc, c0, N, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopJ.getBody());
    Value j = loopJ.getInductionVar();

    auto elemType = outType.getElementType();
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
    rewriter.create<memref::StoreOp>(loc, zero, out, ValueRange{i, j});

    rewriter.create<scf::ForOp>(
        loc, c0, K, c1, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value k, ValueRange) {
          Value a = b.create<memref::LoadOp>(loc2, lhs, ValueRange{i, k});
          Value b_rhs = b.create<memref::LoadOp>(loc2, rhs, ValueRange{k, j});
          Value cur = b.create<memref::LoadOp>(loc2, out, ValueRange{i, j});
          Value mul = b.create<arith::MulFOp>(loc2, a, b_rhs);
          Value add = b.create<arith::AddFOp>(loc2, cur, mul);
          b.create<memref::StoreOp>(loc2, add, out, ValueRange{i, j});
          b.create<scf::YieldOp>(loc2);
        });

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct AddOpConversion : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value out = adaptor.getOut();

    auto outType = out.getType().cast<MemRefType>();
    if (outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "only 2D tensors supported");

    Location loc = op.getLoc();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, M, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value i = loopI.getInductionVar();

    rewriter.create<scf::ForOp>(
        loc, c0, N, c1, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value j, ValueRange) {
          Value a = b.create<memref::LoadOp>(loc2, lhs, ValueRange{i, j});
          Value b_rhs = b.create<memref::LoadOp>(loc2, rhs, ValueRange{i, j});
          Value add = b.create<arith::AddFOp>(loc2, a, b_rhs);
          b.create<memref::StoreOp>(loc2, add, out, ValueRange{i, j});
          b.create<scf::YieldOp>(loc2);
        });

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ReluOpConversion : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    Value out = adaptor.getOut();

    auto outType = out.getType().cast<MemRefType>();
    if (outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "only 2D tensors supported");

    auto elemType = outType.getElementType();
    Location loc = op.getLoc();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, M, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value i = loopI.getInductionVar();

    rewriter.create<scf::ForOp>(
        loc, c0, N, c1, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value j, ValueRange) {
          Value x = b.create<memref::LoadOp>(loc2, input, ValueRange{i, j});
          Value cmp = b.create<arith::CmpFOp>(loc2, arith::CmpFPredicate::OGT, x, zero);
          Value max = b.create<arith::SelectOp>(loc2, cmp, x, zero);
          b.create<memref::StoreOp>(loc2, max, out, ValueRange{i, j});
          b.create<scf::YieldOp>(loc2);
        });

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct FullFlowOpConversion : public OpConversionPattern<FullFlowOp> {
  using OpConversionPattern<FullFlowOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(FullFlowOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value bias = adaptor.getBias();
    Value out = adaptor.getOut();

    auto outType = out.getType().cast<MemRefType>();
    if (outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "only 2D tensors supported");

    Location loc = op.getLoc();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);
    Value K = rewriter.create<memref::DimOp>(loc, lhs, 1);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto elemType = outType.getElementType();
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, M, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value i = loopI.getInductionVar();

    auto loopJ = rewriter.create<scf::ForOp>(loc, c0, N, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopJ.getBody());
    Value j = loopJ.getInductionVar();

    auto loopK = rewriter.create<scf::ForOp>(
        loc, c0, K, c1, ValueRange{zero},
        [&](OpBuilder &b, Location loc2, Value k, ValueRange iterArgs) {
          Value acc = iterArgs[0];
          Value a = b.create<memref::LoadOp>(loc2, lhs, ValueRange{i, k});
          Value b_rhs = b.create<memref::LoadOp>(loc2, rhs, ValueRange{k, j});
          Value mul = b.create<arith::MulFOp>(loc2, a, b_rhs);
          Value add = b.create<arith::AddFOp>(loc2, acc, mul);
          b.create<scf::YieldOp>(loc2, ValueRange{add});
        });
    Value sum = loopK.getResult(0);

    Value biasVal = rewriter.create<memref::LoadOp>(loc, bias, ValueRange{i, j});
    Value sumPlusBias = rewriter.create<arith::AddFOp>(loc, sum, biasVal);
    Value cmp = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, sumPlusBias, zero);
    Value result = rewriter.create<arith::SelectOp>(loc, cmp, sumPlusBias, zero);
    rewriter.create<memref::StoreOp>(loc, result, out, ValueRange{i, j});

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    bool needsConversion = llvm::any_of(op.getOperands(), [](Value v) {
      return v.getType().isa<RankedTensorType>();
    });
    if (!needsConversion)
      return failure();

    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, operands);

    return success();
  }
};

} // namespace

namespace cinder {

struct FCGPassBaseline : public impl::FCGPassBaselineBase<FCGPassBaseline> {
  using impl::FCGPassBaselineBase<FCGPassBaseline>::FCGPassBaselineBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();

    TypeConverter converter;
    converter.addConversion(convertTensorToMemRef);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalOp<ModuleOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
      auto fnType = func.getFunctionType();
      auto hasTensor = [](Type t) { return t.isa<RankedTensorType>(); };
      return !llvm::any_of(fnType.getInputs(), hasTensor) &&
            !llvm::any_of(fnType.getResults(), hasTensor);
    });

    target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
      for (auto operand : op.getOperands()) {
        if (operand.getType().isa<RankedTensorType>())
          return false;
      }
      return true;
    });

    target.addIllegalDialect<cinder::FCGDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ReturnOpConversion>(converter, patterns.getContext());
    patterns.add<MatMulOpConversion>(converter, patterns.getContext());
    patterns.add<AddOpConversion>(converter, patterns.getContext());
    patterns.add<ReluOpConversion>(converter, patterns.getContext());
    patterns.add<FullFlowOpConversion>(converter, patterns.getContext());

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, converter);

    if (failed(applyFullConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createFCGPassBaseline() {
  return std::make_unique<FCGPassBaseline>();
}

} // namespace cinder
