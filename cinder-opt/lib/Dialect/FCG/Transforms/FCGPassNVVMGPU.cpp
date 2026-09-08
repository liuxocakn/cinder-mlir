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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

#include "cinder/Dialect/FCG/FCG.h"

#define GEN_PASS_DEF_FCGPASSNVVMGPU
#include "cinder/Dialect/FCG/FCGPasses.h"

using namespace mlir;
using namespace cinder;

namespace {

struct MatMulOpGen : private MatMulOp {
  MLIRContext *ctx;
  Location loc;
  OpBuilder builder;
  MatMulOpGen(ModuleOp op) : ctx(op.getContext()), loc(op.getLoc()), builder(op.getContext()) {
    kernel(op, getKernelNameF32(), Float32Type::get(ctx));
    kernel(op, getKernelNameF16(), Float16Type::get(ctx));
  }
private:
  void kernel(ModuleOp op, StringRef kernelName, Type elementType) {
    builder.setInsertionPointToStart(op.getBody());

    auto ptrType = LLVM::LLVMPointerType::get(elementType, 1);
    auto i64Type = IntegerType::get(ctx, 64);
    auto i32Type = IntegerType::get(ctx, 32);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    SmallVector<Type, 21> paramTypes;
    for (int i = 0; i < 3; ++i) {
      paramTypes.push_back(ptrType);
      paramTypes.push_back(ptrType);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
    }

    auto funcType = LLVM::LLVMFunctionType::get(voidType, paramTypes, false);

    auto funcOp = builder.create<LLVM::LLVMFuncOp>(loc, kernelName, funcType);
    funcOp->setAttr("nvvm.kernel", UnitAttr::get(ctx));

    const char* names[] = {
      "A_data",   "A_aligned",   "A_offset",   "A_size0",   "A_size1",   "A_stride0",   "A_stride1",
      "B_data",   "B_aligned",   "B_offset",   "B_size0",   "B_size1",   "B_stride0",   "B_stride1",
      "out_data", "out_aligned", "out_offset", "out_size0", "out_size1", "out_stride0", "out_stride1"
    };
    for (unsigned i = 0; i < 21; ++i) {
      funcOp.setArgAttr(i, "llvm.name", StringAttr::get(ctx, names[i]));
    }

    Block *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto getArg = [&](unsigned idx) -> Value {
      return funcOp.getArgument(idx);
    };

    Value A_data    = getArg(0);
    Value A_aligned = getArg(1);
    Value A_offset  = getArg(2);
    Value A_size0   = getArg(3);
    Value A_size1   = getArg(4);
    Value A_stride0 = getArg(5);
    Value A_stride1 = getArg(6);
    (void)A_data;

    Value B_data    = getArg(7);
    Value B_aligned = getArg(8);
    Value B_offset  = getArg(9);
    Value B_size0   = getArg(10);
    Value B_size1   = getArg(11);
    Value B_stride0 = getArg(12);
    Value B_stride1 = getArg(13);
    (void)B_data;
    (void)B_size0;

    Value out_data    = getArg(14);
    Value out_aligned = getArg(15);
    Value out_offset  = getArg(16);
    Value out_size0   = getArg(17);
    Value out_size1   = getArg(18);
    Value out_stride0 = getArg(19);
    Value out_stride1 = getArg(20);
    (void)out_data;
    (void)out_size0;
    (void)out_size1;
    (void)out_stride1;

    Value M = A_size0;
    Value N = B_size1;
    Value K = A_size1;

    auto tidX_i32 = builder.create<NVVM::ThreadIdXOp>(loc, i32Type);
    auto tidY_i32 = builder.create<NVVM::ThreadIdYOp>(loc, i32Type);
    auto bidX_i32 = builder.create<NVVM::BlockIdXOp>(loc, i32Type);
    auto bidY_i32 = builder.create<NVVM::BlockIdYOp>(loc, i32Type);
    auto dimX_i32 = builder.create<NVVM::BlockDimXOp>(loc, i32Type);
    auto dimY_i32 = builder.create<NVVM::BlockDimYOp>(loc, i32Type);

    auto tidX = builder.create<arith::ExtUIOp>(loc, i64Type, tidX_i32);
    auto tidY = builder.create<arith::ExtUIOp>(loc, i64Type, tidY_i32);
    auto bidX = builder.create<arith::ExtUIOp>(loc, i64Type, bidX_i32);
    auto bidY = builder.create<arith::ExtUIOp>(loc, i64Type, bidY_i32);
    auto dimX = builder.create<arith::ExtUIOp>(loc, i64Type, dimX_i32);
    auto dimY = builder.create<arith::ExtUIOp>(loc, i64Type, dimY_i32);

    auto rowMul = builder.create<LLVM::MulOp>(loc, i64Type, bidY, dimY);
    auto colMul = builder.create<LLVM::MulOp>(loc, i64Type, bidX, dimX);
    Value row   = builder.create<LLVM::AddOp>(loc, i64Type, rowMul, tidY);
    Value col   = builder.create<LLVM::AddOp>(loc, i64Type, colMul, tidX);

    auto cmpRow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row, M);
    auto cmpCol = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, col, N);
    auto cond   = builder.create<arith::AndIOp>(loc, cmpRow, cmpCol);

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, false);
    Block *thenBlock = ifOp.thenBlock();
    builder.setInsertionPointToStart(thenBlock);

    bool isF16 = elementType.isF16();
    Type accType = isF16 ? Float32Type::get(ctx) : elementType;
    auto zeroAttr = builder.getFloatAttr(accType, 0.0);
    Value init = builder.create<arith::ConstantOp>(loc, zeroAttr);

    Value zeroIdx = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0));
    Value oneIdx  = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1));

    auto forOp = builder.create<scf::ForOp>(loc, zeroIdx, K, oneIdx, ValueRange{init});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(forOp.getBody());
      Value k = forOp.getInductionVar();
      Value acc = forOp.getRegionIterArg(0);

      Value idxA = builder.create<LLVM::MulOp>(loc, i64Type, row, A_stride0);
      Value kA   = builder.create<LLVM::MulOp>(loc, i64Type, k, A_stride1);
      idxA = builder.create<LLVM::AddOp>(loc, i64Type, idxA, kA);
      idxA = builder.create<LLVM::AddOp>(loc, i64Type, idxA, A_offset);

      Value idxB = builder.create<LLVM::MulOp>(loc, i64Type, k, B_stride0);
      Value cB   = builder.create<LLVM::MulOp>(loc, i64Type, col, B_stride1);
      idxB = builder.create<LLVM::AddOp>(loc, i64Type, idxB, cB);
      idxB = builder.create<LLVM::AddOp>(loc, i64Type, idxB, B_offset);

      auto gepA = builder.create<LLVM::GEPOp>(loc, ptrType, A_aligned, ValueRange{idxA});
      auto gepB = builder.create<LLVM::GEPOp>(loc, ptrType, B_aligned, ValueRange{idxB});

      auto valA = builder.create<LLVM::LoadOp>(loc, elementType, gepA);
      auto valB = builder.create<LLVM::LoadOp>(loc, elementType, gepB);
      Value mul;
      if (isF16) {
        auto f32Type = Float32Type::get(ctx);
        auto valA_f = builder.create<arith::ExtFOp>(loc, f32Type, valA);
        auto valB_f = builder.create<arith::ExtFOp>(loc, f32Type, valB);
        mul = builder.create<arith::MulFOp>(loc, valA_f, valB_f);
      } else {
        mul = builder.create<arith::MulFOp>(loc, valA, valB);
      }
      Value newAcc = builder.create<arith::AddFOp>(loc, acc, mul);

      builder.create<scf::YieldOp>(loc, ValueRange{newAcc});
    }

    Value sum = forOp.getResult(0);
    if (isF16) {
      auto f16Type = Float16Type::get(ctx);
      sum = builder.create<arith::TruncFOp>(loc, f16Type, sum);
    }

    Value idxOut = builder.create<LLVM::MulOp>(loc, i64Type, row, out_stride0);
    idxOut = builder.create<LLVM::AddOp>(loc, i64Type, idxOut, col);
    idxOut = builder.create<LLVM::AddOp>(loc, i64Type, idxOut, out_offset);

    auto gepOut = builder.create<LLVM::GEPOp>(loc, ptrType, out_aligned, ValueRange{idxOut});
    builder.create<LLVM::StoreOp>(loc, sum, gepOut);

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }
};

struct AddOpGen : private AddOp {
  MLIRContext *ctx;
  Location loc;
  OpBuilder builder;
  AddOpGen(ModuleOp op) : ctx(op.getContext()), loc(op.getLoc()), builder(op.getContext()) {
    kernel(op, getKernelNameF32(), Float32Type::get(ctx));
    kernel(op, getKernelNameF16(), Float16Type::get(ctx));
  }
private:
  void kernel(ModuleOp op, StringRef kernelName, Type elementType) {
    builder.setInsertionPointToStart(op.getBody());

    auto ptrType = LLVM::LLVMPointerType::get(elementType, 1);
    auto i64Type = IntegerType::get(ctx, 64);
    auto i32Type = IntegerType::get(ctx, 32);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    SmallVector<Type, 21> paramTypes;
    for (int i = 0; i < 3; ++i) {
      paramTypes.push_back(ptrType);
      paramTypes.push_back(ptrType);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
    }

    auto funcType = LLVM::LLVMFunctionType::get(voidType, paramTypes, false);

    auto funcOp = builder.create<LLVM::LLVMFuncOp>(loc, kernelName, funcType);
    funcOp->setAttr("nvvm.kernel", UnitAttr::get(ctx));

    const char* names[] = {
      "A_data",   "A_aligned",   "A_offset",   "A_size0",   "A_size1",   "A_stride0",   "A_stride1",
      "B_data",   "B_aligned",   "B_offset",   "B_size0",   "B_size1",   "B_stride0",   "B_stride1",
      "out_data", "out_aligned", "out_offset", "out_size0", "out_size1", "out_stride0", "out_stride1"
    };
    for (unsigned i = 0; i < 21; ++i) {
      funcOp.setArgAttr(i, "llvm.name", StringAttr::get(ctx, names[i]));
    }

    Block *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto getArg = [&](unsigned idx) -> Value {
      return funcOp.getArgument(idx);
    };

    Value A_data    = getArg(0);
    Value A_aligned = getArg(1);
    Value A_offset  = getArg(2);
    Value A_size0   = getArg(3);
    Value A_size1   = getArg(4);
    Value A_stride0 = getArg(5);
    Value A_stride1 = getArg(6);
    (void)A_data;
    (void)A_stride1;

    Value B_data    = getArg(7);
    Value B_aligned = getArg(8);
    Value B_offset  = getArg(9);
    Value B_size0   = getArg(10);
    Value B_size1   = getArg(11);
    Value B_stride0 = getArg(12);
    Value B_stride1 = getArg(13);
    (void)B_data;
    (void)B_size0;
    (void)B_size1;
    (void)B_stride1;

    Value out_data    = getArg(14);
    Value out_aligned = getArg(15);
    Value out_offset  = getArg(16);
    Value out_size0   = getArg(17);
    Value out_size1   = getArg(18);
    Value out_stride0 = getArg(19);
    Value out_stride1 = getArg(20);
    (void)out_data;
    (void)out_size0;
    (void)out_size1;
    (void)out_stride1;

    Value M = A_size0;
    Value N = A_size1;

    auto tidX_i32 = builder.create<NVVM::ThreadIdXOp>(loc, i32Type);
    auto tidY_i32 = builder.create<NVVM::ThreadIdYOp>(loc, i32Type);
    auto bidX_i32 = builder.create<NVVM::BlockIdXOp>(loc, i32Type);
    auto bidY_i32 = builder.create<NVVM::BlockIdYOp>(loc, i32Type);
    auto dimX_i32 = builder.create<NVVM::BlockDimXOp>(loc, i32Type);
    auto dimY_i32 = builder.create<NVVM::BlockDimYOp>(loc, i32Type);

    auto tidX = builder.create<arith::ExtUIOp>(loc, i64Type, tidX_i32);
    auto tidY = builder.create<arith::ExtUIOp>(loc, i64Type, tidY_i32);
    auto bidX = builder.create<arith::ExtUIOp>(loc, i64Type, bidX_i32);
    auto bidY = builder.create<arith::ExtUIOp>(loc, i64Type, bidY_i32);
    auto dimX = builder.create<arith::ExtUIOp>(loc, i64Type, dimX_i32);
    auto dimY = builder.create<arith::ExtUIOp>(loc, i64Type, dimY_i32);

    auto rowMul = builder.create<LLVM::MulOp>(loc, i64Type, bidY, dimY);
    auto colMul = builder.create<LLVM::MulOp>(loc, i64Type, bidX, dimX);
    Value row   = builder.create<LLVM::AddOp>(loc, i64Type, rowMul, tidY);
    Value col   = builder.create<LLVM::AddOp>(loc, i64Type, colMul, tidX);

    auto cmpRow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row, M);
    auto cmpCol = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, col, N);
    auto cond   = builder.create<arith::AndIOp>(loc, cmpRow, cmpCol);

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, false);
    Block *thenBlock = ifOp.thenBlock();
    builder.setInsertionPointToStart(thenBlock);

    Value idxA = builder.create<LLVM::MulOp>(loc, i64Type, row, A_stride0);
    idxA = builder.create<LLVM::AddOp>(loc, i64Type, idxA, col);
    idxA = builder.create<LLVM::AddOp>(loc, i64Type, idxA, A_offset);

    Value idxB = builder.create<LLVM::MulOp>(loc, i64Type, row, B_stride0);
    idxB = builder.create<LLVM::AddOp>(loc, i64Type, idxB, col);
    idxB = builder.create<LLVM::AddOp>(loc, i64Type, idxB, B_offset);

    Value idxOut = builder.create<LLVM::MulOp>(loc, i64Type, row, out_stride0);
    idxOut = builder.create<LLVM::AddOp>(loc, i64Type, idxOut, col);
    idxOut = builder.create<LLVM::AddOp>(loc, i64Type, idxOut, out_offset);

    auto gepA = builder.create<LLVM::GEPOp>(loc, ptrType, A_aligned, ValueRange{idxA});
    auto gepB = builder.create<LLVM::GEPOp>(loc, ptrType, B_aligned, ValueRange{idxB});
    auto gepOut = builder.create<LLVM::GEPOp>(loc, ptrType, out_aligned, ValueRange{idxOut});

    auto valA = builder.create<LLVM::LoadOp>(loc, elementType, gepA);
    auto valB = builder.create<LLVM::LoadOp>(loc, elementType, gepB);
    auto sum = builder.create<arith::AddFOp>(loc, valA, valB);
    builder.create<LLVM::StoreOp>(loc, sum, gepOut);

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }
};

struct ReluOpGen : private ReluOp {
  MLIRContext *ctx;
  Location loc;
  OpBuilder builder;
  ReluOpGen(ModuleOp op) : ctx(op.getContext()), loc(op.getLoc()), builder(op.getContext()) {
    kernel(op, getKernelNameF32(), Float32Type::get(ctx));
    kernel(op, getKernelNameF16(), Float16Type::get(ctx));
  }
private:
  void kernel(ModuleOp op, StringRef kernelName, Type elementType) {
    builder.setInsertionPointToStart(op.getBody());

    auto ptrType = LLVM::LLVMPointerType::get(elementType, 1);
    auto i64Type = IntegerType::get(ctx, 64);
    auto i32Type = IntegerType::get(ctx, 32);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    SmallVector<Type, 14> paramTypes;
    for (int i = 0; i < 2; ++i) {
      paramTypes.push_back(ptrType);
      paramTypes.push_back(ptrType);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
    }

    auto funcType = LLVM::LLVMFunctionType::get(voidType, paramTypes, false);

    auto funcOp = builder.create<LLVM::LLVMFuncOp>(loc, kernelName, funcType);
    funcOp->setAttr("nvvm.kernel", UnitAttr::get(ctx));

    const char* names[] = {
      "input_data",   "input_aligned",   "input_offset",   "input_size0",   "input_size1",   "input_stride0",   "input_stride1",
      "out_data",     "out_aligned",     "out_offset",     "out_size0",     "out_size1",     "out_stride0",     "out_stride1"
    };
    for (unsigned i = 0; i < 14; ++i) {
      funcOp.setArgAttr(i, "llvm.name", StringAttr::get(ctx, names[i]));
    }

    Block *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto getArg = [&](unsigned idx) -> Value {
      return funcOp.getArgument(idx);
    };

    Value input_data    = getArg(0);
    Value input_aligned = getArg(1);
    Value input_offset  = getArg(2);
    Value input_size0   = getArg(3);
    Value input_size1   = getArg(4);
    Value input_stride0 = getArg(5);
    Value input_stride1 = getArg(6);
    (void)input_data;
    (void)input_stride1;

    Value out_data    = getArg(7);
    Value out_aligned = getArg(8);
    Value out_offset  = getArg(9);
    Value out_size0   = getArg(10);
    Value out_size1   = getArg(11);
    Value out_stride0 = getArg(12);
    Value out_stride1 = getArg(13);
    (void)out_data;
    (void)out_size0;
    (void)out_size1;
    (void)out_stride1;

    Value M = input_size0;
    Value N = input_size1;

    auto tidX_i32 = builder.create<NVVM::ThreadIdXOp>(loc, i32Type);
    auto tidY_i32 = builder.create<NVVM::ThreadIdYOp>(loc, i32Type);
    auto bidX_i32 = builder.create<NVVM::BlockIdXOp>(loc, i32Type);
    auto bidY_i32 = builder.create<NVVM::BlockIdYOp>(loc, i32Type);
    auto dimX_i32 = builder.create<NVVM::BlockDimXOp>(loc, i32Type);
    auto dimY_i32 = builder.create<NVVM::BlockDimYOp>(loc, i32Type);

    auto tidX = builder.create<arith::ExtUIOp>(loc, i64Type, tidX_i32);
    auto tidY = builder.create<arith::ExtUIOp>(loc, i64Type, tidY_i32);
    auto bidX = builder.create<arith::ExtUIOp>(loc, i64Type, bidX_i32);
    auto bidY = builder.create<arith::ExtUIOp>(loc, i64Type, bidY_i32);
    auto dimX = builder.create<arith::ExtUIOp>(loc, i64Type, dimX_i32);
    auto dimY = builder.create<arith::ExtUIOp>(loc, i64Type, dimY_i32);

    auto rowMul = builder.create<LLVM::MulOp>(loc, i64Type, bidY, dimY);
    auto colMul = builder.create<LLVM::MulOp>(loc, i64Type, bidX, dimX);
    Value row   = builder.create<LLVM::AddOp>(loc, i64Type, rowMul, tidY);
    Value col   = builder.create<LLVM::AddOp>(loc, i64Type, colMul, tidX);

    auto cmpRow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row, M);
    auto cmpCol = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, col, N);
    auto cond   = builder.create<arith::AndIOp>(loc, cmpRow, cmpCol);

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, false);
    Block *thenBlock = ifOp.thenBlock();
    builder.setInsertionPointToStart(thenBlock);

    Value idxIn = builder.create<LLVM::MulOp>(loc, i64Type, row, input_stride0);
    idxIn = builder.create<LLVM::AddOp>(loc, i64Type, idxIn, col);
    idxIn = builder.create<LLVM::AddOp>(loc, i64Type, idxIn, input_offset);

    auto gepIn = builder.create<LLVM::GEPOp>(loc, ptrType, input_aligned, ValueRange{idxIn});
    auto val = builder.create<LLVM::LoadOp>(loc, elementType, gepIn);

    auto zeroAttr = builder.getFloatAttr(elementType, 0.0);
    Value zero = builder.create<arith::ConstantOp>(loc, zeroAttr);
    auto cmp = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, val, zero);
    Value relu = builder.create<arith::SelectOp>(loc, cmp, val, zero);

    Value idxOut = builder.create<LLVM::MulOp>(loc, i64Type, row, out_stride0);
    idxOut = builder.create<LLVM::AddOp>(loc, i64Type, idxOut, col);
    idxOut = builder.create<LLVM::AddOp>(loc, i64Type, idxOut, out_offset);

    auto gepOut = builder.create<LLVM::GEPOp>(loc, ptrType, out_aligned, ValueRange{idxOut});
    builder.create<LLVM::StoreOp>(loc, relu, gepOut);

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }
};

struct FullFlowOpGen : private FullFlowOp {
  MLIRContext *ctx;
  Location loc;
  OpBuilder builder;
  FullFlowOpGen(ModuleOp op) : ctx(op.getContext()), loc(op.getLoc()), builder(op.getContext()) {
    kernel(op, getKernelNameF32(), Float32Type::get(ctx));
    kernel(op, getKernelNameF16(), Float16Type::get(ctx));
  }
private:
  void kernel(ModuleOp op, StringRef kernelName, Type elementType) {
    builder.setInsertionPointToStart(op.getBody());

    auto ptrType = LLVM::LLVMPointerType::get(elementType, 1);
    auto i64Type = IntegerType::get(ctx, 64);
    auto i32Type = IntegerType::get(ctx, 32);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    SmallVector<Type, 28> paramTypes;
    for (int i = 0; i < 4; ++i) {
      paramTypes.push_back(ptrType);
      paramTypes.push_back(ptrType);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
      paramTypes.push_back(i64Type);
    }

    auto funcType = LLVM::LLVMFunctionType::get(voidType, paramTypes, false);

    auto funcOp = builder.create<LLVM::LLVMFuncOp>(loc, kernelName, funcType);
    funcOp->setAttr("nvvm.kernel", UnitAttr::get(ctx));

    const char* names[] = {
      "A_data",   "A_aligned",   "A_offset",   "A_size0",   "A_size1",   "A_stride0",   "A_stride1",
      "B_data",   "B_aligned",   "B_offset",   "B_size0",   "B_size1",   "B_stride0",   "B_stride1",
      "bias_data","bias_aligned","bias_offset","bias_size0","bias_size1","bias_stride0","bias_stride1",
      "out_data", "out_aligned", "out_offset", "out_size0", "out_size1", "out_stride0", "out_stride1"
    };
    for (unsigned i = 0; i < 28; ++i) {
      funcOp.setArgAttr(i, "llvm.name", StringAttr::get(ctx, names[i]));
    }

    Block *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto getArg = [&](unsigned idx) -> Value {
      return funcOp.getArgument(idx);
    };

    Value A_data    = getArg(0);
    Value A_aligned = getArg(1);
    Value A_offset  = getArg(2);
    Value A_size0   = getArg(3);
    Value A_size1   = getArg(4);
    Value A_stride0 = getArg(5);
    Value A_stride1 = getArg(6);
    (void)A_data;

    Value B_data    = getArg(7);
    Value B_aligned = getArg(8);
    Value B_offset  = getArg(9);
    Value B_size0   = getArg(10);
    Value B_size1   = getArg(11);
    Value B_stride0 = getArg(12);
    Value B_stride1 = getArg(13);
    (void)B_data;
    (void)B_size0;

    Value bias_data    = getArg(14);
    Value bias_aligned = getArg(15);
    Value bias_offset  = getArg(16);
    Value bias_size0   = getArg(17);
    Value bias_size1   = getArg(18);
    Value bias_stride0 = getArg(19);
    Value bias_stride1 = getArg(20);
    (void)bias_data;
    (void)bias_size0;
    (void)bias_size1;
    (void)bias_stride1;

    Value out_data    = getArg(21);
    Value out_aligned = getArg(22);
    Value out_offset  = getArg(23);
    Value out_size0   = getArg(24);
    Value out_size1   = getArg(25);
    Value out_stride0 = getArg(26);
    Value out_stride1 = getArg(27);
    (void)out_data;
    (void)out_stride1;

    Value M = A_size0;
    Value N = B_size1;
    Value K = A_size1;
    (void)M;
    (void)N;

    auto tidX_i32 = builder.create<NVVM::ThreadIdXOp>(loc, i32Type);
    auto tidY_i32 = builder.create<NVVM::ThreadIdYOp>(loc, i32Type);
    auto bidX_i32 = builder.create<NVVM::BlockIdXOp>(loc, i32Type);
    auto bidY_i32 = builder.create<NVVM::BlockIdYOp>(loc, i32Type);
    auto dimX_i32 = builder.create<NVVM::BlockDimXOp>(loc, i32Type);
    auto dimY_i32 = builder.create<NVVM::BlockDimYOp>(loc, i32Type);

    auto tidX = builder.create<arith::ExtUIOp>(loc, i64Type, tidX_i32);
    auto tidY = builder.create<arith::ExtUIOp>(loc, i64Type, tidY_i32);
    auto bidX = builder.create<arith::ExtUIOp>(loc, i64Type, bidX_i32);
    auto bidY = builder.create<arith::ExtUIOp>(loc, i64Type, bidY_i32);
    auto dimX = builder.create<arith::ExtUIOp>(loc, i64Type, dimX_i32);
    auto dimY = builder.create<arith::ExtUIOp>(loc, i64Type, dimY_i32);

    auto rowMul = builder.create<LLVM::MulOp>(loc, i64Type, bidY, dimY);
    auto colMul = builder.create<LLVM::MulOp>(loc, i64Type, bidX, dimX);
    Value row   = builder.create<LLVM::AddOp>(loc, i64Type, rowMul, tidY);
    Value col   = builder.create<LLVM::AddOp>(loc, i64Type, colMul, tidX);

    auto cmpRow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, row, out_size0);
    auto cmpCol = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, col, out_size1);
    auto cond   = builder.create<arith::AndIOp>(loc, cmpRow, cmpCol);

    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, false);
    Block *thenBlock = ifOp.thenBlock();
    builder.setInsertionPointToStart(thenBlock);

    bool isF16 = elementType.isF16();
    Type accType = isF16 ? Float32Type::get(ctx) : elementType;
    auto zeroAttr = builder.getFloatAttr(accType, 0.0);
    Value init = builder.create<arith::ConstantOp>(loc, zeroAttr);

    Value zeroIdx = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0));
    Value oneIdx  = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1));

    auto forOp = builder.create<scf::ForOp>(loc, zeroIdx, K, oneIdx, ValueRange{init});
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(forOp.getBody());
      Value k = forOp.getInductionVar();
      Value acc = forOp.getRegionIterArg(0);

      Value idxA = builder.create<LLVM::MulOp>(loc, i64Type, row, A_stride0);
      Value kA   = builder.create<LLVM::MulOp>(loc, i64Type, k, A_stride1);
      idxA = builder.create<LLVM::AddOp>(loc, i64Type, idxA, kA);
      idxA = builder.create<LLVM::AddOp>(loc, i64Type, idxA, A_offset);

      Value idxB = builder.create<LLVM::MulOp>(loc, i64Type, k, B_stride0);
      Value cB   = builder.create<LLVM::MulOp>(loc, i64Type, col, B_stride1);
      idxB = builder.create<LLVM::AddOp>(loc, i64Type, idxB, cB);
      idxB = builder.create<LLVM::AddOp>(loc, i64Type, idxB, B_offset);

      auto gepA = builder.create<LLVM::GEPOp>(loc, ptrType, A_aligned, ValueRange{idxA});
      auto gepB = builder.create<LLVM::GEPOp>(loc, ptrType, B_aligned, ValueRange{idxB});

      auto valA = builder.create<LLVM::LoadOp>(loc, elementType, gepA);
      auto valB = builder.create<LLVM::LoadOp>(loc, elementType, gepB);

      Value newAcc;
      if (isF16) {
        auto f32Type = Float32Type::get(ctx);
        auto valA_f = builder.create<arith::ExtFOp>(loc, f32Type, valA);
        auto valB_f = builder.create<arith::ExtFOp>(loc, f32Type, valB);
        newAcc = builder.create<LLVM::FMulAddOp>(loc, f32Type, valA_f, valB_f, acc);
      } else {
        auto mul = builder.create<arith::MulFOp>(loc, elementType, valA, valB);
        newAcc = builder.create<arith::AddFOp>(loc, elementType, acc, mul);
      }

      builder.create<scf::YieldOp>(loc, ValueRange{newAcc});
    }

    Value sum = forOp.getResult(0);
    Type f32Type = Float32Type::get(ctx);

    Value idxBias = builder.create<LLVM::MulOp>(loc, i64Type, row, bias_stride0);
    idxBias = builder.create<LLVM::AddOp>(loc, i64Type, idxBias, col);
    idxBias = builder.create<LLVM::AddOp>(loc, i64Type, idxBias, bias_offset);

    auto gepBias = builder.create<LLVM::GEPOp>(loc, ptrType, bias_aligned, ValueRange{idxBias});
    auto biasVal = builder.create<LLVM::LoadOp>(loc, elementType, gepBias);
    Value biasValF;
    if (isF16) {
      biasValF = builder.create<arith::ExtFOp>(loc, f32Type, biasVal);
    } else {
      biasValF = biasVal;
    }
    Value sumWithBias = builder.create<arith::AddFOp>(loc, isF16 ? f32Type : elementType, sum, biasValF);

    Type cmpType = isF16 ? f32Type : elementType;
    auto zeroAttrCmp = builder.getFloatAttr(cmpType, 0.0);
    Value zero = builder.create<arith::ConstantOp>(loc, zeroAttrCmp);
    auto cmp = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, sumWithBias, zero);
    Value relu = builder.create<arith::SelectOp>(loc, cmpType, cmp, sumWithBias, zero);

    Value storeVal = relu;
    if (isF16) {
      storeVal = builder.create<arith::TruncFOp>(loc, elementType, relu);
    }

    Value idxOut = builder.create<LLVM::MulOp>(loc, i64Type, row, out_stride0);
    idxOut = builder.create<LLVM::AddOp>(loc, i64Type, idxOut, col);
    idxOut = builder.create<LLVM::AddOp>(loc, i64Type, idxOut, out_offset);

    auto gepOut = builder.create<LLVM::GEPOp>(loc, ptrType, out_aligned, ValueRange{idxOut});
    builder.create<LLVM::StoreOp>(loc, storeVal, gepOut);

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }
};

} // namespace

namespace cinder {

struct FCGPassNVVMGPU : public impl::FCGPassNVVMGPUBase<FCGPassNVVMGPU> {

  using impl::FCGPassNVVMGPUBase<FCGPassNVVMGPU>::FCGPassNVVMGPUBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp topModule;
    if (needSkip(topModule)) {
      return;
    }

    llvm::DenseSet<mlir::TypeID> presentTypeIDs;
    topModule.walk([&](Operation *op) {
      presentTypeIDs.insert(op->getRegisteredInfo()->getTypeID());
    });

    topModule.getBody()->clear();
    gen_kernel(topModule, presentTypeIDs);
  }

private:
  StringRef topModuleName = "fcg-pass-nvvm-gpu";

  bool needSkip(ModuleOp &topModule) {
    if (getOperation()->getParentOfType<ModuleOp>()) {
      return true;
    }

    topModule = getOperation();
    if (topModule.getName() == topModuleName) {
      return true;
    }

    topModule.setName(topModuleName);
    return false;
  }

  void gen_kernel(ModuleOp &topModule, llvm::DenseSet<mlir::TypeID> &presentTypeIDs) {
    static llvm::DenseMap<mlir::TypeID, std::function<void(ModuleOp)>> generatorMap = {
      {TypeID::get<MatMulOp>(),   [](ModuleOp m){ MatMulOpGen{m}; }},
      {TypeID::get<AddOp>(),      [](ModuleOp m){ AddOpGen{m}; }},
      {TypeID::get<ReluOp>(),     [](ModuleOp m){ ReluOpGen{m}; }},
      {TypeID::get<FullFlowOp>(), [](ModuleOp m){ FullFlowOpGen{m}; }},
    };

    for (auto typeID : presentTypeIDs) {
      auto it = generatorMap.find(typeID);
      if (it != generatorMap.end())
        it->second(topModule);
    }
  }
};

std::unique_ptr<mlir::Pass> createFCGPassNVVMGPU() {
  return std::make_unique<FCGPassNVVMGPU>();
}

} // namespace cinder
