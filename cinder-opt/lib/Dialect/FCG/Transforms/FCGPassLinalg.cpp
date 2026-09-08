#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "cinder/Dialect/FCG/FCG.h"

#define GEN_PASS_DEF_FCGPASSLINALG
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

    auto lhsType = lhs.getType().dyn_cast<MemRefType>();
    auto rhsType = rhs.getType().dyn_cast<MemRefType>();
    auto outType = out.getType().dyn_cast<MemRefType>();
    if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "requires 2D memrefs");
    }

    AffineMap lhsMap = AffineMap::get(3, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(2)}, rewriter.getContext());
    AffineMap rhsMap = AffineMap::get(3, 0, {rewriter.getAffineDimExpr(2), rewriter.getAffineDimExpr(1)}, rewriter.getContext());
    AffineMap outMap = AffineMap::get(3, 0, {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)}, rewriter.getContext());
    SmallVector<AffineMap> maps = {lhsMap, rhsMap, outMap};

    SmallVector<utils::IteratorType> iterTypes = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel,
        utils::IteratorType::reduction
    };

    rewriter.create<linalg::GenericOp>(
        op.getLoc(),
        ValueRange{lhs, rhs},
        ValueRange{out},
        maps,
        iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value add = b.create<arith::AddFOp>(loc, args[2], mul);
          b.create<linalg::YieldOp>(loc, add);
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

    auto lhsType = lhs.getType().dyn_cast<MemRefType>();
    auto rhsType = rhs.getType().dyn_cast<MemRefType>();
    auto outType = out.getType().dyn_cast<MemRefType>();
    if (!lhsType || !rhsType || !outType) {
      return rewriter.notifyMatchFailure(op, "requires memref types");
    }

    int outRank = outType.getRank();
    int lhsRank = lhsType.getRank();
    int rhsRank = rhsType.getRank();

    SmallVector<utils::IteratorType> iteratorTypes(outRank, utils::IteratorType::parallel);

    AffineMap outMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());

    SmallVector<AffineExpr> lhsExprs;
    for (int i = 0; i < outRank; ++i) {
      if (i < lhsRank) {
        int outDim = outRank - lhsRank + i;
        lhsExprs.push_back(rewriter.getAffineDimExpr(outDim));
      } else {
        lhsExprs.push_back(rewriter.getAffineConstantExpr(0));
      }
    }
    AffineMap lhsMap = AffineMap::get(outRank, 0, lhsExprs, rewriter.getContext());

    SmallVector<AffineExpr> rhsExprs;
    for (int i = 0; i < outRank; ++i) {
      if (i < rhsRank) {
        int outDim = outRank - rhsRank + i;
        rhsExprs.push_back(rewriter.getAffineDimExpr(outDim));
      } else {
        rhsExprs.push_back(rewriter.getAffineConstantExpr(0));
      }
    }
    AffineMap rhsMap = AffineMap::get(outRank, 0, rhsExprs, rewriter.getContext());

    SmallVector<AffineMap> maps = {lhsMap, rhsMap, outMap};

    rewriter.create<linalg::GenericOp>(
        op.getLoc(),
        ValueRange{lhs, rhs},
        ValueRange{out},
        maps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value add = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, add);
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

    auto inputType = input.getType().dyn_cast<MemRefType>();
    auto outType = out.getType().dyn_cast<MemRefType>();
    if (!inputType || !outType || inputType.getRank() != outType.getRank()) {
      return rewriter.notifyMatchFailure(op, "requires memrefs with same rank");
    }

    int rank = outType.getRank();
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineMap> maps = {identityMap, identityMap};

    rewriter.create<linalg::GenericOp>(
        op.getLoc(),
        ValueRange{input},
        ValueRange{out},
        maps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(x.getType()));
          Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, x, zero);
          Value max = b.create<arith::SelectOp>(loc, cmp, x, zero);
          b.create<linalg::YieldOp>(loc, max);
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

    auto outType = out.getType().dyn_cast<MemRefType>();
    if (!outType || outType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "requires 2D memref for out");

    rewriter.create<linalg::MatmulOp>(op.getLoc(), ValueRange{lhs, rhs}, ValueRange{out});

    int rank = outType.getRank();
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineMap> maps = {identityMap, identityMap, identityMap};

    rewriter.create<linalg::GenericOp>(
        op.getLoc(),
        ValueRange{out, bias},
        ValueRange{out},
        maps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value mulResult = args[0];
          Value biasVal = args[1];
          Value sum = b.create<arith::AddFOp>(loc, mulResult, biasVal);
          Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(sum.getType()));
          Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, sum, zero);
          Value relu = b.create<arith::SelectOp>(loc, cmp, sum, zero);
          b.create<linalg::YieldOp>(loc, relu);
        });

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

struct FCGPassLinalg : public impl::FCGPassLinalgBase<FCGPassLinalg> {
  using impl::FCGPassLinalgBase<FCGPassLinalg>::FCGPassLinalgBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<linalg::LinalgDialect>();
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
    target.addLegalDialect<linalg::LinalgDialect>();
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

std::unique_ptr<mlir::Pass> createFCGPassLinalg() {
  return std::make_unique<FCGPassLinalg>();
}

} // namespace cinder
