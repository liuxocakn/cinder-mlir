#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "cinder/Dialect/FCG/FCG.h"

#define GEN_PASS_DEF_FCGPASSVECTORIZE
#include "cinder/Dialect/FCG/FCGPasses.h"

using namespace mlir;
using namespace cinder;

static constexpr int64_t VECTOR_WIDTH = 32;

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

    auto elemType = outType.getElementType();
    if (!elemType.isF32() && !elemType.isF16())
      return rewriter.notifyMatchFailure(op, "only f32/f16 supported");

    Location loc = op.getLoc();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);
    Value K = rewriter.create<memref::DimOp>(loc, lhs, 1);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value vectorWidth = rewriter.create<arith::ConstantIndexOp>(loc, VECTOR_WIDTH);

    VectorType vecType = VectorType::get(VECTOR_WIDTH, elemType);
    Value zeroVec = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(vecType));

    AffineMap colMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)}, rewriter.getContext());

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, M, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value i = loopI.getInductionVar();

    Value nRem = rewriter.create<arith::RemUIOp>(loc, N, vectorWidth);
    Value nVec = rewriter.create<arith::SubIOp>(loc, N, nRem);

    rewriter.create<scf::ForOp>(
        loc, c0, nVec, vectorWidth, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value j, ValueRange) {
          Value acc = zeroVec;

          auto loopK = b.create<scf::ForOp>(
              loc2, c0, K, c1, ValueRange{acc},
              [&](OpBuilder &kb, Location kloc, Value k, ValueRange args) {
                Value accIn = args[0];
                Value aScalar = kb.create<memref::LoadOp>(kloc, lhs, ValueRange{i, k});
                Value bVec = kb.create<vector::TransferReadOp>(
                    kloc, vecType, rhs, ValueRange{k, j}, colMap);
                Value aVec = kb.create<vector::BroadcastOp>(kloc, vecType, aScalar);
                Value mulVec = kb.create<arith::MulFOp>(kloc, aVec, bVec);
                Value addVec = kb.create<arith::AddFOp>(kloc, accIn, mulVec);
                kb.create<scf::YieldOp>(kloc, addVec);
              });
          acc = loopK.getResult(0);

          b.create<vector::TransferWriteOp>(loc2, acc, out, ValueRange{i, j}, colMap);
          b.create<scf::YieldOp>(loc2);
        });

    rewriter.create<scf::ForOp>(
        loc, nVec, N, c1, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value j, ValueRange) {
          Value accScalar = b.create<arith::ConstantOp>(loc2, rewriter.getZeroAttr(elemType));
          auto loopK = b.create<scf::ForOp>(
              loc2, c0, K, c1, ValueRange{accScalar},
              [&](OpBuilder &kb, Location kloc, Value k, ValueRange args) {
                Value accIn = args[0];
                Value a = kb.create<memref::LoadOp>(kloc, lhs, ValueRange{i, k});
                Value bval = kb.create<memref::LoadOp>(kloc, rhs, ValueRange{k, j});
                Value mul = kb.create<arith::MulFOp>(kloc, a, bval);
                Value add = kb.create<arith::AddFOp>(kloc, accIn, mul);
                kb.create<scf::YieldOp>(kloc, add);
              });
          accScalar = loopK.getResult(0);
          b.create<memref::StoreOp>(loc2, accScalar, out, ValueRange{i, j});
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

    auto elemType = outType.getElementType();
    if (!elemType.isF32() && !elemType.isF16())
      return rewriter.notifyMatchFailure(op, "only f32/f16 supported");

    Location loc = op.getLoc();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value vectorWidth = rewriter.create<arith::ConstantIndexOp>(loc, VECTOR_WIDTH);

    VectorType vecType = VectorType::get(VECTOR_WIDTH, elemType);
    AffineMap colMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)}, rewriter.getContext());

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, M, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value i = loopI.getInductionVar();

    Value nRem = rewriter.create<arith::RemUIOp>(loc, N, vectorWidth);
    Value nVec = rewriter.create<arith::SubIOp>(loc, N, nRem);

    rewriter.create<scf::ForOp>(
        loc, c0, nVec, vectorWidth, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value j, ValueRange) {
          Value aVec = b.create<vector::TransferReadOp>(
              loc2, vecType, lhs, ValueRange{i, j}, colMap);
          Value bVec = b.create<vector::TransferReadOp>(
              loc2, vecType, rhs, ValueRange{i, j}, colMap);
          Value addVec = b.create<arith::AddFOp>(loc2, aVec, bVec);
          b.create<vector::TransferWriteOp>(loc2, addVec, out, ValueRange{i, j}, colMap);
          b.create<scf::YieldOp>(loc2);
        });

    rewriter.create<scf::ForOp>(
        loc, nVec, N, c1, ValueRange{},
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
    if (!elemType.isF32() && !elemType.isF16())
      return rewriter.notifyMatchFailure(op, "only f32/f16 supported");

    Location loc = op.getLoc();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value vectorWidth = rewriter.create<arith::ConstantIndexOp>(loc, VECTOR_WIDTH);

    VectorType vecType = VectorType::get(VECTOR_WIDTH, elemType);
    Value zeroVec = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(vecType));
    Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
    AffineMap colMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)}, rewriter.getContext());

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, M, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value i = loopI.getInductionVar();

    Value nRem = rewriter.create<arith::RemUIOp>(loc, N, vectorWidth);
    Value nVec = rewriter.create<arith::SubIOp>(loc, N, nRem);

    rewriter.create<scf::ForOp>(
        loc, c0, nVec, vectorWidth, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value j, ValueRange) {
          Value xVec = b.create<vector::TransferReadOp>(
              loc2, vecType, input, ValueRange{i, j}, colMap);
          Value cmpVec = b.create<arith::CmpFOp>(
              loc2, arith::CmpFPredicate::OGT, xVec, zeroVec);
          Value maxVec = b.create<arith::SelectOp>(loc2, cmpVec, xVec, zeroVec);
          b.create<vector::TransferWriteOp>(loc2, maxVec, out, ValueRange{i, j}, colMap);
          b.create<scf::YieldOp>(loc2);
        });

    rewriter.create<scf::ForOp>(
        loc, nVec, N, c1, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value j, ValueRange) {
          Value x = b.create<memref::LoadOp>(loc2, input, ValueRange{i, j});
          Value cmp = b.create<arith::CmpFOp>(loc2, arith::CmpFPredicate::OGT, x, zeroScalar);
          Value max = b.create<arith::SelectOp>(loc2, cmp, x, zeroScalar);
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

    auto elemType = outType.getElementType();
    if (!elemType.isF32() && !elemType.isF16())
      return rewriter.notifyMatchFailure(op, "only f32/f16 supported");

    Location loc = op.getLoc();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);
    Value K = rewriter.create<memref::DimOp>(loc, lhs, 1);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value vectorWidth = rewriter.create<arith::ConstantIndexOp>(loc, VECTOR_WIDTH);

    VectorType vecType = VectorType::get(VECTOR_WIDTH, elemType);
    Value zeroVec = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(vecType));
    Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));

    AffineMap colMap = AffineMap::get(2, 0, {rewriter.getAffineDimExpr(1)}, rewriter.getContext());

    auto loopI = rewriter.create<scf::ForOp>(loc, c0, M, c1, ValueRange{});
    rewriter.setInsertionPointToStart(loopI.getBody());
    Value i = loopI.getInductionVar();

    Value nRem = rewriter.create<arith::RemUIOp>(loc, N, vectorWidth);
    Value nVec = rewriter.create<arith::SubIOp>(loc, N, nRem);

    rewriter.create<scf::ForOp>(
        loc, c0, nVec, vectorWidth, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value j, ValueRange) {
          Value acc = zeroVec;

          auto loopK = b.create<scf::ForOp>(
              loc2, c0, K, c1, ValueRange{acc},
              [&](OpBuilder &kb, Location kloc, Value k, ValueRange args) {
                Value accIn = args[0];
                Value aScalar = kb.create<memref::LoadOp>(kloc, lhs, ValueRange{i, k});
                Value bVec = kb.create<vector::TransferReadOp>(
                    kloc, vecType, rhs, ValueRange{k, j}, colMap);
                Value aVec = kb.create<vector::BroadcastOp>(kloc, vecType, aScalar);
                Value mulVec = kb.create<arith::MulFOp>(kloc, aVec, bVec);
                Value addVec = kb.create<arith::AddFOp>(kloc, accIn, mulVec);
                kb.create<scf::YieldOp>(kloc, addVec);
              });
          acc = loopK.getResult(0);

          Value biasVec = b.create<vector::TransferReadOp>(
              loc2, vecType, bias, ValueRange{i, j}, colMap);
          Value sumVec = b.create<arith::AddFOp>(loc2, acc, biasVec);
          Value cmpVec = b.create<arith::CmpFOp>(loc2, arith::CmpFPredicate::OGT, sumVec, zeroVec);
          Value reluVec = b.create<arith::SelectOp>(loc2, cmpVec, sumVec, zeroVec);
          b.create<vector::TransferWriteOp>(loc2, reluVec, out, ValueRange{i, j}, colMap);
          b.create<scf::YieldOp>(loc2);
        });

    rewriter.create<scf::ForOp>(
        loc, nVec, N, c1, ValueRange{},
        [&](OpBuilder &b, Location loc2, Value j, ValueRange) {
          Value accScalar = zeroScalar;
          auto loopK = b.create<scf::ForOp>(
              loc2, c0, K, c1, ValueRange{accScalar},
              [&](OpBuilder &kb, Location kloc, Value k, ValueRange args) {
                Value accIn = args[0];
                Value a = kb.create<memref::LoadOp>(kloc, lhs, ValueRange{i, k});
                Value bval = kb.create<memref::LoadOp>(kloc, rhs, ValueRange{k, j});
                Value mul = kb.create<arith::MulFOp>(kloc, a, bval);
                Value add = kb.create<arith::AddFOp>(kloc, accIn, mul);
                kb.create<scf::YieldOp>(kloc, add);
              });
          accScalar = loopK.getResult(0);

          Value biasScalar = b.create<memref::LoadOp>(loc2, bias, ValueRange{i, j});
          Value sumScalar = b.create<arith::AddFOp>(loc2, accScalar, biasScalar);
          Value cmpScalar = b.create<arith::CmpFOp>(loc2, arith::CmpFPredicate::OGT, sumScalar, zeroScalar);
          Value reluScalar = b.create<arith::SelectOp>(loc2, cmpScalar, sumScalar, zeroScalar);
          b.create<memref::StoreOp>(loc2, reluScalar, out, ValueRange{i, j});
          b.create<scf::YieldOp>(loc2);
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

struct FCGPassVectorize : public impl::FCGPassVectorizeBase<FCGPassVectorize> {
  using impl::FCGPassVectorizeBase<FCGPassVectorize>::FCGPassVectorizeBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<vector::VectorDialect>();
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
    target.addLegalDialect<vector::VectorDialect>();
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

std::unique_ptr<mlir::Pass> createFCGPassVectorize() {
  return std::make_unique<FCGPassVectorize>();
}

} // namespace cinder
