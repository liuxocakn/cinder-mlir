#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "cinder/Dialect/FCG/FCG.h"

#define GEN_PASS_DEF_FCGPASSGPU
#include "cinder/Dialect/FCG/FCGPasses.h"

using namespace mlir;
using namespace cinder;

namespace {

static Type convertTensorToMemRef(Type type) {
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    auto addrSpace = IntegerAttr::get(IntegerType::get(tensorType.getContext(), 64), 1);
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                           MemRefLayoutAttrInterface{}, addrSpace);
  }
  return type;
}

static gpu::GPUFuncOp createGPUKernel(PatternRewriter &rewriter, Location loc,
                                       StringRef kernelName,
                                       ArrayRef<Type> argTypes,
                                       function_ref<void(OpBuilder &, Location, Block *)> bodyBuilder) {
  auto moduleOp = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  if (!moduleOp->hasAttr("gpu.container_module")) {
    moduleOp->setAttr("gpu.container_module", UnitAttr::get(rewriter.getContext()));
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto gpuModule = rewriter.create<gpu::GPUModuleOp>(loc, kernelName);
  gpuModule->setAttr("gpu.binary", StringAttr::get(rewriter.getContext(), ""));

  auto ctx = rewriter.getContext();
  auto funcType = FunctionType::get(ctx, argTypes, TypeRange{});
  OpBuilder kernelBuilder(gpuModule.getBodyRegion());
  auto kernelFunc = kernelBuilder.create<gpu::GPUFuncOp>(
      loc, "kernel", funcType,
      TypeRange{},
      TypeRange{});
  kernelFunc->setAttr("gpu.kernel", UnitAttr::get(ctx));

  Block *entryBlock = nullptr;
  if (kernelFunc.getBody().empty()) {
    entryBlock = kernelFunc.addEntryBlock();
  } else {
    entryBlock = &kernelFunc.getBody().front();
  }

  OpBuilder bodyBuilderObj(ctx);
  bodyBuilderObj.setInsertionPointToStart(entryBlock);
  bodyBuilder(bodyBuilderObj, loc, entryBlock);
  bodyBuilderObj.create<gpu::ReturnOp>(loc);

  return kernelFunc;
}

static void createLaunchFunc(PatternRewriter &rewriter, Location loc,
                             gpu::GPUFuncOp kernelFunc,
                             ArrayRef<Value> inputs, Value output,
                             Value gridSize, Value blockSize) {
  SmallVector<Value> operands;
  operands.append(inputs.begin(), inputs.end());
  operands.push_back(output);

  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  gpu::KernelDim3 gridDim = {gridSize, one, one};
  gpu::KernelDim3 blockDim = {blockSize, one, one};

  rewriter.create<gpu::LaunchFuncOp>(
      loc,
      kernelFunc,
      gridDim,
      blockDim,
      nullptr,
      operands,
      nullptr,
      ValueRange{});
}

static Value computeGridSize(PatternRewriter &rewriter, Location loc, Value totalElems, int64_t blockSize) {
  Value blockSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  return rewriter.create<arith::DivUIOp>(loc,
               rewriter.create<arith::AddIOp>(loc,
                 rewriter.create<arith::SubIOp>(loc, totalElems, one),
                 blockSizeVal),
               blockSizeVal);
}

struct MatMulOpConversion : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto out = adaptor.getOut();

    auto elemType = lhs.getType().cast<MemRefType>().getElementType();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);
    Value totalElems = rewriter.create<arith::MulIOp>(loc, M, N);
    const int64_t blockSize = 256;
    Value blockSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
    Value gridSize = computeGridSize(rewriter, loc, totalElems, blockSize);

    SmallVector<Type> argTypes = {lhs.getType(), rhs.getType(), out.getType()};
    static int kernelCounter = 0;
    std::string kernelName = "matmul_kernel_" + std::to_string(kernelCounter++);
    auto kernelFunc = createGPUKernel(rewriter, loc, kernelName, argTypes,
      [&](OpBuilder &b, Location loc, Block *block) {
        auto args = block->getArguments();
        Value lhsVal = args[0];
        Value rhsVal = args[1];
        Value outVal = args[2];

        Value M_k = b.create<memref::DimOp>(loc, outVal, 0);
        Value N_k = b.create<memref::DimOp>(loc, outVal, 1);
        Value K = b.create<memref::DimOp>(loc, lhsVal, 1);

        Value blockIdx = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
        Value threadIdx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
        Value blockDim = b.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
        Value globalId = b.create<arith::AddIOp>(loc,
                          b.create<arith::MulIOp>(loc, blockIdx, blockDim),
                          threadIdx);

        Value total = b.create<arith::MulIOp>(loc, M_k, N_k);
        Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 globalId, total);
        b.create<scf::IfOp>(loc, inBounds,
          [&](OpBuilder &thenBuilder, Location thenLoc) {
            Value i = thenBuilder.create<arith::DivUIOp>(thenLoc, globalId, N_k);
            Value j = thenBuilder.create<arith::RemUIOp>(thenLoc, globalId, N_k);

            Value zero = thenBuilder.create<arith::ConstantOp>(thenLoc,
                         thenBuilder.getZeroAttr(elemType));
            Value zeroIdx = thenBuilder.create<arith::ConstantIndexOp>(thenLoc, 0);
            Value oneIdx = thenBuilder.create<arith::ConstantIndexOp>(thenLoc, 1);
            auto loop = thenBuilder.create<scf::ForOp>(thenLoc, zeroIdx, K, oneIdx,
                                                       ValueRange({zero}),
              [&](OpBuilder &loopBuilder, Location loopLoc, Value iv, ValueRange iterArgs) {
                Value sumVal = iterArgs[0];
                SmallVector<Value, 2> idxA = {i, iv};
                SmallVector<Value, 2> idxB = {iv, j};
                Value a = loopBuilder.create<memref::LoadOp>(loopLoc, lhsVal, idxA);
                Value bVal = loopBuilder.create<memref::LoadOp>(loopLoc, rhsVal, idxB);
                Value mul = loopBuilder.create<arith::MulFOp>(loopLoc, a, bVal);
                Value newSum = loopBuilder.create<arith::AddFOp>(loopLoc, sumVal, mul);
                loopBuilder.create<scf::YieldOp>(loopLoc, ValueRange({newSum}));
              });
            Value result = loop.getResult(0);
            SmallVector<Value, 2> idxOut = {i, j};
            thenBuilder.create<memref::StoreOp>(thenLoc, result, outVal, idxOut);
            thenBuilder.create<scf::YieldOp>(thenLoc);
          },
          [&](OpBuilder &elseBuilder, Location elseLoc) {
            elseBuilder.create<scf::YieldOp>(elseLoc);
          }
        );
      }
    );

    createLaunchFunc(rewriter, loc, kernelFunc, {lhs, rhs}, out, gridSize, blockSizeVal);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct AddOpConversion : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto out = adaptor.getOut();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);
    Value totalElems = rewriter.create<arith::MulIOp>(loc, M, N);
    const int64_t blockSize = 256;
    Value blockSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
    Value gridSize = computeGridSize(rewriter, loc, totalElems, blockSize);

    SmallVector<Type> argTypes = {lhs.getType(), rhs.getType(), out.getType()};
    static int kernelCounter = 0;
    std::string kernelName = "add_kernel_" + std::to_string(kernelCounter++);
    auto kernelFunc = createGPUKernel(rewriter, loc, kernelName, argTypes,
      [&](OpBuilder &b, Location loc, Block *block) {
        auto args = block->getArguments();
        Value lhsVal = args[0];
        Value rhsVal = args[1];
        Value outVal = args[2];

        Value M_k = b.create<memref::DimOp>(loc, outVal, 0);
        Value N_k = b.create<memref::DimOp>(loc, outVal, 1);

        Value blockIdx = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
        Value threadIdx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
        Value blockDim = b.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
        Value globalId = b.create<arith::AddIOp>(loc,
                          b.create<arith::MulIOp>(loc, blockIdx, blockDim),
                          threadIdx);

        Value total = b.create<arith::MulIOp>(loc, M_k, N_k);
        Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 globalId, total);
        b.create<scf::IfOp>(loc, inBounds,
          [&](OpBuilder &thenBuilder, Location thenLoc) {
            Value i = thenBuilder.create<arith::DivUIOp>(thenLoc, globalId, N_k);
            Value j = thenBuilder.create<arith::RemUIOp>(thenLoc, globalId, N_k);
            SmallVector<Value, 2> idx = {i, j};
            Value lhsElem = thenBuilder.create<memref::LoadOp>(thenLoc, lhsVal, idx);
            Value rhsElem = thenBuilder.create<memref::LoadOp>(thenLoc, rhsVal, idx);
            Value sum = thenBuilder.create<arith::AddFOp>(thenLoc, lhsElem, rhsElem);
            thenBuilder.create<memref::StoreOp>(thenLoc, sum, outVal, idx);
            thenBuilder.create<scf::YieldOp>(thenLoc);
          },
          [&](OpBuilder &elseBuilder, Location elseLoc) {
            elseBuilder.create<scf::YieldOp>(elseLoc);
          }
        );
      }
    );

    createLaunchFunc(rewriter, loc, kernelFunc, {lhs, rhs}, out, gridSize, blockSizeVal);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ReluOpConversion : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto out = adaptor.getOut();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);
    Value totalElems = rewriter.create<arith::MulIOp>(loc, M, N);
    const int64_t blockSize = 256;
    Value blockSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
    Value gridSize = computeGridSize(rewriter, loc, totalElems, blockSize);

    SmallVector<Type> argTypes = {input.getType(), out.getType()};
    static int kernelCounter = 0;
    std::string kernelName = "relu_kernel_" + std::to_string(kernelCounter++);
    auto kernelFunc = createGPUKernel(rewriter, loc, kernelName, argTypes,
      [&](OpBuilder &b, Location loc, Block *block) {
        auto args = block->getArguments();
        Value inputVal = args[0];
        Value outVal = args[1];

        Value M_k = b.create<memref::DimOp>(loc, outVal, 0);
        Value N_k = b.create<memref::DimOp>(loc, outVal, 1);

        Value blockIdx = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
        Value threadIdx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
        Value blockDim = b.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
        Value globalId = b.create<arith::AddIOp>(loc,
                          b.create<arith::MulIOp>(loc, blockIdx, blockDim),
                          threadIdx);

        Value total = b.create<arith::MulIOp>(loc, M_k, N_k);
        Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 globalId, total);
        b.create<scf::IfOp>(loc, inBounds,
          [&](OpBuilder &thenBuilder, Location thenLoc) {
            Value i = thenBuilder.create<arith::DivUIOp>(thenLoc, globalId, N_k);
            Value j = thenBuilder.create<arith::RemUIOp>(thenLoc, globalId, N_k);
            SmallVector<Value, 2> idx = {i, j};
            Value x = thenBuilder.create<memref::LoadOp>(thenLoc, inputVal, idx);
            Value zero = thenBuilder.create<arith::ConstantOp>(thenLoc,
              thenBuilder.getZeroAttr(x.getType()));
            Value cmp = thenBuilder.create<arith::CmpFOp>(thenLoc, arith::CmpFPredicate::OGT, x, zero);
            Value maxVal = thenBuilder.create<arith::SelectOp>(thenLoc, cmp, x, zero);
            thenBuilder.create<memref::StoreOp>(thenLoc, maxVal, outVal, idx);
            thenBuilder.create<scf::YieldOp>(thenLoc);
          },
          [&](OpBuilder &elseBuilder, Location elseLoc) {
            elseBuilder.create<scf::YieldOp>(elseLoc);
          }
        );
      }
    );

    createLaunchFunc(rewriter, loc, kernelFunc, {input}, out, gridSize, blockSizeVal);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct FullFlowOpConversion : public OpConversionPattern<FullFlowOp> {
  using OpConversionPattern<FullFlowOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(FullFlowOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto bias = adaptor.getBias();
    auto out = adaptor.getOut();

    auto elemType = lhs.getType().cast<MemRefType>().getElementType();

    Value M = rewriter.create<memref::DimOp>(loc, out, 0);
    Value N = rewriter.create<memref::DimOp>(loc, out, 1);
    Value totalElems = rewriter.create<arith::MulIOp>(loc, M, N);
    const int64_t blockSize = 256;
    Value blockSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);
    Value gridSize = computeGridSize(rewriter, loc, totalElems, blockSize);

    SmallVector<Type> argTypes = {lhs.getType(), rhs.getType(), bias.getType(), out.getType()};
    static int kernelCounter = 0;
    std::string kernelName = "fullflow_kernel_" + std::to_string(kernelCounter++);
    auto kernelFunc = createGPUKernel(rewriter, loc, kernelName, argTypes,
      [&](OpBuilder &b, Location loc, Block *block) {
        auto args = block->getArguments();
        Value lhsVal = args[0];
        Value rhsVal = args[1];
        Value biasVal = args[2];
        Value outVal = args[3];

        Value M_k = b.create<memref::DimOp>(loc, outVal, 0);
        Value N_k = b.create<memref::DimOp>(loc, outVal, 1);
        Value K = b.create<memref::DimOp>(loc, lhsVal, 1);

        Value blockIdx = b.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
        Value threadIdx = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
        Value blockDim = b.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
        Value globalId = b.create<arith::AddIOp>(loc,
                          b.create<arith::MulIOp>(loc, blockIdx, blockDim),
                          threadIdx);

        Value total = b.create<arith::MulIOp>(loc, M_k, N_k);
        Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 globalId, total);
        b.create<scf::IfOp>(loc, inBounds,
          [&](OpBuilder &thenBuilder, Location thenLoc) {
            Value i = thenBuilder.create<arith::DivUIOp>(thenLoc, globalId, N_k);
            Value j = thenBuilder.create<arith::RemUIOp>(thenLoc, globalId, N_k);

            Value zero = thenBuilder.create<arith::ConstantOp>(thenLoc,
                         thenBuilder.getZeroAttr(elemType));
            Value zeroIdx = thenBuilder.create<arith::ConstantIndexOp>(thenLoc, 0);
            Value oneIdx = thenBuilder.create<arith::ConstantIndexOp>(thenLoc, 1);
            auto loop = thenBuilder.create<scf::ForOp>(thenLoc, zeroIdx, K, oneIdx,
                                                       ValueRange({zero}),
              [&](OpBuilder &loopBuilder, Location loopLoc, Value iv, ValueRange iterArgs) {
                Value sumVal = iterArgs[0];
                SmallVector<Value, 2> idxA = {i, iv};
                SmallVector<Value, 2> idxB = {iv, j};
                Value a = loopBuilder.create<memref::LoadOp>(loopLoc, lhsVal, idxA);
                Value bVal = loopBuilder.create<memref::LoadOp>(loopLoc, rhsVal, idxB);
                Value mul = loopBuilder.create<arith::MulFOp>(loopLoc, a, bVal);
                Value newSum = loopBuilder.create<arith::AddFOp>(loopLoc, sumVal, mul);
                loopBuilder.create<scf::YieldOp>(loopLoc, ValueRange({newSum}));
              });
            Value matmulRes = loop.getResult(0);

            SmallVector<Value, 2> biasIdx = {i, j};
            Value biasElem = thenBuilder.create<memref::LoadOp>(thenLoc, biasVal, biasIdx);
            Value addRes = thenBuilder.create<arith::AddFOp>(thenLoc, matmulRes, biasElem);

            Value zeroVal = thenBuilder.create<arith::ConstantOp>(thenLoc,
                            thenBuilder.getZeroAttr(elemType));
            Value cmp = thenBuilder.create<arith::CmpFOp>(thenLoc, arith::CmpFPredicate::OGT, addRes, zeroVal);
            Value reluRes = thenBuilder.create<arith::SelectOp>(thenLoc, cmp, addRes, zeroVal);

            SmallVector<Value, 2> idxOut = {i, j};
            thenBuilder.create<memref::StoreOp>(thenLoc, reluRes, outVal, idxOut);
            thenBuilder.create<scf::YieldOp>(thenLoc);
          },
          [&](OpBuilder &elseBuilder, Location elseLoc) {
            elseBuilder.create<scf::YieldOp>(elseLoc);
          }
        );
      }
    );

    createLaunchFunc(rewriter, loc, kernelFunc, {lhs, rhs, bias}, out, gridSize, blockSizeVal);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

} // namespace

namespace cinder {

struct FCGPassGPU : public impl::FCGPassGPUBase<FCGPassGPU> {
  using impl::FCGPassGPUBase<FCGPassGPU>::FCGPassGPUBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();

    if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
      if (!moduleOp->hasAttr("gpu.container_module")) {
        moduleOp->setAttr("gpu.container_module", UnitAttr::get(&getContext()));
      }
    }

    TypeConverter converter;
    converter.addConversion(convertTensorToMemRef);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<vector::VectorDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalOp<ModuleOp, gpu::ReturnOp, gpu::GPUModuleOp, gpu::GPUFuncOp, gpu::LaunchFuncOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();

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

std::unique_ptr<mlir::Pass> createFCGPassGPU() {
  return std::make_unique<FCGPassGPU>();
}

} // namespace cinder
