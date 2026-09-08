#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MemoryBuffer.h"

#include "cinder/Dialect/FCG/FCG.h"

#define GEN_PASS_DEF_FCGPASSNVVMCPU
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

static constexpr const char *prefix = "fcg_pass_nvvm_cpu";

static void genCUDARuntime(ModuleOp topModule, StringRef kernelPath, ArrayRef<StringRef> kernelNames) {
  auto loc = topModule.getLoc();
  OpBuilder builder(topModule.getContext());

  builder.setInsertionPointToStart(topModule.getBody());

  auto i32Type    = builder.getI32Type();
  auto i64Type    = builder.getI64Type();
  auto i8Type     = builder.getI8Type();
  auto voidType   = LLVM::LLVMVoidType::get(builder.getContext());
  auto i8PtrType  = LLVM::LLVMPointerType::get(i8Type, 0);
  auto i32PtrType = LLVM::LLVMPointerType::get(i32Type, 0);
  auto i64PtrType = LLVM::LLVMPointerType::get(i64Type, 0);

  auto fileOrErr = llvm::MemoryBuffer::getFile(kernelPath);
  if (auto EC = fileOrErr.getError()) {
    topModule.emitError("Failed to read kernel cubin file: " + kernelPath + " - " + EC.message());
    return;
  }
  std::string cubinData = (*fileOrErr)->getBuffer().str();
  auto cubinGlobal = builder.create<LLVM::GlobalOp>(loc,
                      LLVM::LLVMArrayType::get(i8Type, cubinData.size()),
                      true,
                      LLVM::Linkage::Internal,
                      ".cubin_data",
                      StringAttr::get(builder.getContext(),
                                      StringRef(cubinData.data(), cubinData.size())));

  auto cuInitTy                = LLVM::LLVMFunctionType::get(i32Type, {i32Type});
  auto cuDeviceGetTy           = LLVM::LLVMFunctionType::get(i32Type, {i32PtrType, i32Type});
  auto cuCtxCreateTy           = LLVM::LLVMFunctionType::get(i32Type, {i64PtrType, i32Type, i32Type});
  auto cuModuleLoadDataTy      = LLVM::LLVMFunctionType::get(i32Type, {i64PtrType, i8PtrType});
  auto cuModuleGetFunctionTy   = LLVM::LLVMFunctionType::get(i32Type, {i64PtrType, i64Type, i8PtrType});
  auto cuLaunchKernelTy        = LLVM::LLVMFunctionType::get(i32Type, {i64Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i8PtrType, i8PtrType});
  auto cuCtxSynchronizeTy      = LLVM::LLVMFunctionType::get(i32Type, {});
  auto cuCtxDestroyTy          = LLVM::LLVMFunctionType::get(i32Type, {i64Type});

  builder.create<LLVM::LLVMFuncOp>(loc, "cuInit",                cuInitTy,                LLVM::Linkage::External);
  builder.create<LLVM::LLVMFuncOp>(loc, "cuDeviceGet",           cuDeviceGetTy,           LLVM::Linkage::External);
  builder.create<LLVM::LLVMFuncOp>(loc, "cuCtxCreate_v2",        cuCtxCreateTy,           LLVM::Linkage::External);
  builder.create<LLVM::LLVMFuncOp>(loc, "cuModuleLoadData",      cuModuleLoadDataTy,      LLVM::Linkage::External);
  builder.create<LLVM::LLVMFuncOp>(loc, "cuModuleGetFunction",   cuModuleGetFunctionTy,   LLVM::Linkage::External);
  builder.create<LLVM::LLVMFuncOp>(loc, "cuLaunchKernel",        cuLaunchKernelTy,        LLVM::Linkage::External);
  builder.create<LLVM::LLVMFuncOp>(loc, "cuCtxSynchronize",      cuCtxSynchronizeTy,      LLVM::Linkage::External);
  builder.create<LLVM::LLVMFuncOp>(loc, "cuCtxDestroy",          cuCtxDestroyTy,          LLVM::Linkage::External);

  std::string cudaHandleName = (std::string(prefix) + "_cuda_handle");
  auto cudaHandle = builder.create<LLVM::GlobalOp>(loc,
                    i64Type,
                    false,
                    LLVM::Linkage::External,
                    cudaHandleName,
                    builder.getIntegerAttr(i64Type, 0)
  );

  for (const auto &name : kernelNames) {
    builder.create<LLVM::GlobalOp>(loc,
                        i64Type,
                        false,
                        LLVM::Linkage::External,
                        name,
                        builder.getIntegerAttr(i64Type, 0)
    );
  }

  for (const auto &name : kernelNames) {
    std::string nameValue = name.str();
    nameValue.push_back('\0');
    builder.create<LLVM::GlobalOp>(loc,
                          LLVM::LLVMArrayType::get(i8Type, nameValue.size()),
                          true,
                          LLVM::Linkage::Internal,
                          (name + "_str").str(),
                          StringAttr::get(
                            builder.getContext(),
                            StringRef(nameValue.data(), nameValue.size())
                          )
    );
  }

  builder.setInsertionPointToStart(topModule.getBody());
  std::string initFuncName = (std::string(prefix) + "_gpu_init");
  auto initFunc = builder.create<LLVM::LLVMFuncOp>(loc,
    initFuncName,
    LLVM::LLVMFunctionType::get(voidType, {}),
    LLVM::Linkage::External
  );

  auto &initEntry = initFunc.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&initEntry);

  auto cudaHandlePtrCtor = builder.create<LLVM::AddressOfOp>(loc, LLVM::LLVMPointerType::get(i64Type, 0), cudaHandle.getName());

  auto globalType = cubinGlobal.getType();
  auto ptrType = LLVM::LLVMPointerType::get(globalType, 0);
  auto cubinAddr = builder.create<LLVM::AddressOfOp>(loc, ptrType, cubinGlobal.getName());
  auto cubin = builder.create<LLVM::BitcastOp>(loc, i8PtrType, cubinAddr);

  auto zeroI32 = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0));
  builder.create<LLVM::CallOp>(loc, i32Type, "cuInit", ValueRange{zeroI32});

  auto devicePtr = builder.create<LLVM::AllocaOp>(loc, i32PtrType, zeroI32, 0);
  builder.create<LLVM::CallOp>(loc, i32Type, "cuDeviceGet", ValueRange{devicePtr, zeroI32});

  auto deviceVal = builder.create<LLVM::LoadOp>(loc, i32Type, devicePtr);
  auto ctxPtr = builder.create<LLVM::AllocaOp>(loc, i64PtrType, zeroI32, 0);
  builder.create<LLVM::CallOp>(loc, i32Type, "cuCtxCreate_v2", ValueRange{ctxPtr, zeroI32, deviceVal});

  auto ctxVal = builder.create<LLVM::LoadOp>(loc, i64Type, ctxPtr);
  builder.create<LLVM::StoreOp>(loc, ctxVal, cudaHandlePtrCtor);

  auto modulePtr = builder.create<LLVM::AllocaOp>(loc, i64PtrType, zeroI32, 0);
  builder.create<LLVM::CallOp>(loc, i32Type, "cuModuleLoadData", ValueRange{modulePtr, cubin});
  auto modVal = builder.create<LLVM::LoadOp>(loc, i64Type, modulePtr);

  for (const auto &name : kernelNames) {
    auto handleGlobalAddr = builder.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(i64Type, 0), name);
    auto strGlobalArray = builder.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(i8Type, name.size() + 1), 0), (name + "_str").str());
    auto strGlobal = builder.create<LLVM::BitcastOp>(loc, i8PtrType, strGlobalArray);
    auto funcPtr = builder.create<LLVM::AllocaOp>(loc, i64PtrType, zeroI32, 0);
    builder.create<LLVM::CallOp>(loc, i32Type, "cuModuleGetFunction", ValueRange{funcPtr, modVal, strGlobal});
    auto funcVal = builder.create<LLVM::LoadOp>(loc, i64Type, funcPtr);
    builder.create<LLVM::StoreOp>(loc, funcVal, handleGlobalAddr);
  }

  builder.create<LLVM::ReturnOp>(loc, ValueRange{});

  builder.setInsertionPointToStart(topModule.getBody());
  std::string finiFuncName = (std::string(prefix) + "_gpu_fini");
  auto finiFunc = builder.create<LLVM::LLVMFuncOp>(loc,
    finiFuncName,
    LLVM::LLVMFunctionType::get(voidType, {}),
    LLVM::Linkage::Internal
  );

  auto &dtorEntry = finiFunc.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&dtorEntry);

  auto cudaHandlePtrDtor = builder.create<LLVM::AddressOfOp>(loc, LLVM::LLVMPointerType::get(i64Type, 0), cudaHandle.getName());
  auto ctxLoad = builder.create<LLVM::LoadOp>(loc, i64Type, cudaHandlePtrDtor);
  builder.create<LLVM::CallOp>(loc, i32Type, "cuCtxDestroy", ValueRange{ctxLoad});
  builder.create<LLVM::ReturnOp>(loc, ValueRange{});

  builder.setInsertionPointToStart(topModule.getBody());
  SmallVector<Attribute> ctorSym = { SymbolRefAttr::get(initFunc) };
  SmallVector<Attribute> ctorPrio = { builder.getI32IntegerAttr(65535) };
  auto ctorArrayAttr = builder.getArrayAttr(ctorSym);
  auto prioArrayAttr = builder.getArrayAttr(ctorPrio);
  builder.create<LLVM::GlobalCtorsOp>(loc, ctorArrayAttr, prioArrayAttr);

  SmallVector<Attribute> dtorSym = { SymbolRefAttr::get(finiFunc) };
  SmallVector<Attribute> dtorPrio = { builder.getI32IntegerAttr(65535) };
  auto dtorArrayAttr = builder.getArrayAttr(dtorSym);
  auto dtorPrioArrayAttr = builder.getArrayAttr(dtorPrio);
  builder.create<LLVM::GlobalDtorsOp>(loc, dtorArrayAttr, dtorPrioArrayAttr);
}

static std::tuple<Value, Value, Value, Value, Value, Value, Value>
genExtractMemRefMetadata(Value memref, ConversionPatternRewriter &rewriter, Location loc) {
  auto memRefType = memref.getType().cast<MemRefType>();
  unsigned rank = memRefType.getRank();
  assert(rank == 2 && "Only 2D memrefs supported");

  auto addrSpace = memRefType.getMemorySpaceAsInt();
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), addrSpace);
  auto i64Type = rewriter.getI64Type();
  auto array2xi64 = LLVM::LLVMArrayType::get(i64Type, 2);

  SmallVector<Type, 5> structTypes = {ptrType, ptrType, i64Type, array2xi64, array2xi64};
  auto structType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), structTypes);

  auto castOp = rewriter.create<UnrealizedConversionCastOp>(loc, structType, memref);
  Value desc = castOp.getResult(0);

  Value base = rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, desc, llvm::ArrayRef<int64_t>{0});
  Value aligned = rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, desc, llvm::ArrayRef<int64_t>{1});
  Value offset = rewriter.create<LLVM::ExtractValueOp>(loc, i64Type, desc, llvm::ArrayRef<int64_t>{2});

  Value size0 = rewriter.create<LLVM::ExtractValueOp>(loc, i64Type, desc, llvm::ArrayRef<int64_t>{3, 0});
  Value size1 = rewriter.create<LLVM::ExtractValueOp>(loc, i64Type, desc, llvm::ArrayRef<int64_t>{3, 1});
  Value stride0 = rewriter.create<LLVM::ExtractValueOp>(loc, i64Type, desc, llvm::ArrayRef<int64_t>{4, 0});
  Value stride1 = rewriter.create<LLVM::ExtractValueOp>(loc, i64Type, desc, llvm::ArrayRef<int64_t>{4, 1});

  auto i8PtrType = LLVM::LLVMPointerType::get(rewriter.getI8Type(), addrSpace);
  Value data = rewriter.create<LLVM::BitcastOp>(loc, i8PtrType, base);

  return {data, aligned, offset, size0, size1, stride0, stride1};
}

static void genCuLaunchKernel(ConversionPatternRewriter &rewriter, Location loc,
                               StringRef kernelName,
                               ArrayRef<Value> params,
                               Value gridX,  Value gridY,  Value gridZ,
                               Value blockX, Value blockY, Value blockZ) {
  auto i64Type = rewriter.getI64Type();
  auto i32Type = rewriter.getI32Type();
  auto i8Type  = rewriter.getI8Type();
  auto i8PtrType = LLVM::LLVMPointerType::get(i8Type, 0);
  auto ptrPtrType = LLVM::LLVMPointerType::get(i8PtrType, 0);

  auto handleGlobal = rewriter.create<LLVM::AddressOfOp>(loc, LLVM::LLVMPointerType::get(i64Type, 0), kernelName);
  auto kernelHandle = rewriter.create<LLVM::LoadOp>(loc, i64Type, handleGlobal);

  SmallVector<Value> paramPtrs;
  for (Value p : params) {
    auto one = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(1));
    auto ptrType = LLVM::LLVMPointerType::get(p.getType(), 0);
    auto alloc = rewriter.create<LLVM::AllocaOp>(loc, ptrType, one, 0);
    rewriter.create<LLVM::StoreOp>(loc, p, alloc);
    paramPtrs.push_back(alloc);
  }

  auto arraySize = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(params.size()));
  auto paramArray = rewriter.create<LLVM::AllocaOp>(loc, ptrPtrType, arraySize, 0);
  for (unsigned i = 0; i < params.size(); ++i) {
    auto idx = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(i));
    auto gep = rewriter.create<LLVM::GEPOp>(loc, ptrPtrType, paramArray, ValueRange{idx});
    auto cast = rewriter.create<LLVM::BitcastOp>(loc, i8PtrType, paramPtrs[i]);
    rewriter.create<LLVM::StoreOp>(loc, cast, gep);
  }
  auto paramArrayCast = rewriter.create<LLVM::BitcastOp>(loc, i8PtrType, paramArray);

  auto zeroI32 = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
  auto zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(0));
  auto nullPtr = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrType, zeroI64);
  rewriter.create<LLVM::CallOp>(loc, i32Type, "cuLaunchKernel",
    ValueRange{
      kernelHandle,
      gridX, gridY, gridZ,
      blockX, blockY, blockZ,
      zeroI32, zeroI32,
      paramArrayCast,
      nullPtr
    }
  );
  rewriter.create<LLVM::CallOp>(loc, i32Type, "cuCtxSynchronize", ValueRange{});
}

struct MatMulOpConversion : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc  = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value out = adaptor.getOut();

    auto [lhsBase, lhsAligned, lhsOffset, lhsSize0, lhsSize1, lhsStride0, lhsStride1] = genExtractMemRefMetadata(lhs, rewriter, loc);
    auto [rhsBase, rhsAligned, rhsOffset, rhsSize0, rhsSize1, rhsStride0, rhsStride1] = genExtractMemRefMetadata(rhs, rewriter, loc);
    auto [outBase, outAligned, outOffset, outSize0, outSize1, outStride0, outStride1] = genExtractMemRefMetadata(out, rewriter, loc);

    auto elementType = out.getType().cast<MemRefType>().getElementType();
    StringRef kernelName;
    if (elementType.isF32()) kernelName = op.getKernelNameF32();
    else if (elementType.isF16()) kernelName = op.getKernelNameF16();
    else return rewriter.notifyMatchFailure(op, "unsupported element type");

    constexpr int blockSize = 32;
    auto i64Type = rewriter.getI64Type();
    auto i32Type = rewriter.getI32Type();
    auto cBlockSizeMinusOne = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(blockSize - 1));
    auto cBlockSize = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(blockSize));
    auto gridX = rewriter.create<LLVM::TruncOp>(loc, i32Type,
                    rewriter.create<LLVM::SDivOp>(loc,
                    rewriter.create<LLVM::AddOp>(loc, outSize0, cBlockSizeMinusOne), cBlockSize));
    auto gridY = rewriter.create<LLVM::TruncOp>(loc, i32Type,
                    rewriter.create<LLVM::SDivOp>(loc,
                    rewriter.create<LLVM::AddOp>(loc, outSize1, cBlockSizeMinusOne), cBlockSize));
    Value gridZ  = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));
    Value blockX = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(blockSize));
    Value blockY = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(blockSize));
    Value blockZ = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));

    SmallVector<Value> params = {
      lhsBase, lhsAligned, lhsOffset, lhsSize0, lhsSize1, lhsStride0, lhsStride1,
      rhsBase, rhsAligned, rhsOffset, rhsSize0, rhsSize1, rhsStride0, rhsStride1,
      outBase, outAligned, outOffset, outSize0, outSize1, outStride0, outStride1
    };

    genCuLaunchKernel(rewriter, loc,
                      kernelName, params,
                      gridX, gridY, gridZ,
                      blockX, blockY, blockZ);

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct AddOpConversion : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc  = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value out = adaptor.getOut();

    auto [lhsBase, lhsAligned, lhsOffset, lhsSize0, lhsSize1, lhsStride0, lhsStride1] = genExtractMemRefMetadata(lhs, rewriter, loc);
    auto [rhsBase, rhsAligned, rhsOffset, rhsSize0, rhsSize1, rhsStride0, rhsStride1] = genExtractMemRefMetadata(rhs, rewriter, loc);
    auto [outBase, outAligned, outOffset, outSize0, outSize1, outStride0, outStride1] = genExtractMemRefMetadata(out, rewriter, loc);

    auto elementType = out.getType().cast<MemRefType>().getElementType();
    StringRef kernelName;
    if (elementType.isF32()) kernelName = op.getKernelNameF32();
    else if (elementType.isF16()) kernelName = op.getKernelNameF16();
    else return rewriter.notifyMatchFailure(op, "unsupported element type");

    constexpr int blockSize = 32;
    auto i64Type = rewriter.getI64Type();
    auto i32Type = rewriter.getI32Type();
    auto cBlockSizeMinusOne = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(blockSize - 1));
    auto cBlockSize = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(blockSize));
    auto gridX = rewriter.create<LLVM::TruncOp>(loc, i32Type,
                    rewriter.create<LLVM::SDivOp>(loc,
                    rewriter.create<LLVM::AddOp>(loc, outSize0, cBlockSizeMinusOne), cBlockSize));
    auto gridY = rewriter.create<LLVM::TruncOp>(loc, i32Type,
                    rewriter.create<LLVM::SDivOp>(loc,
                    rewriter.create<LLVM::AddOp>(loc, outSize1, cBlockSizeMinusOne), cBlockSize));
    Value gridZ  = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));
    Value blockX = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(blockSize));
    Value blockY = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(blockSize));
    Value blockZ = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));

    SmallVector<Value> params = {
      lhsBase, lhsAligned, lhsOffset, lhsSize0, lhsSize1, lhsStride0, lhsStride1,
      rhsBase, rhsAligned, rhsOffset, rhsSize0, rhsSize1, rhsStride0, rhsStride1,
      outBase, outAligned, outOffset, outSize0, outSize1, outStride0, outStride1
    };

    genCuLaunchKernel(rewriter, loc,
                      kernelName, params,
                      gridX, gridY, gridZ,
                      blockX, blockY, blockZ);

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ReluOpConversion : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc   = op.getLoc();
    Value input = adaptor.getInput();
    Value out   = adaptor.getOut();

    auto [inBase, inAligned, inOffset, inSize0, inSize1, inStride0, inStride1] = genExtractMemRefMetadata(input, rewriter, loc);
    auto [outBase, outAligned, outOffset, outSize0, outSize1, outStride0, outStride1] = genExtractMemRefMetadata(out, rewriter, loc);

    auto elementType = out.getType().cast<MemRefType>().getElementType();
    StringRef kernelName;
    if (elementType.isF32()) kernelName = op.getKernelNameF32();
    else if (elementType.isF16()) kernelName = op.getKernelNameF16();
    else return rewriter.notifyMatchFailure(op, "unsupported element type");

    constexpr int blockSize = 32;
    auto i64Type = rewriter.getI64Type();
    auto i32Type = rewriter.getI32Type();
    auto cBlockSizeMinusOne = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(blockSize - 1));
    auto cBlockSize = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(blockSize));
    auto gridX = rewriter.create<LLVM::TruncOp>(loc, i32Type,
                    rewriter.create<LLVM::SDivOp>(loc,
                    rewriter.create<LLVM::AddOp>(loc, outSize0, cBlockSizeMinusOne), cBlockSize));
    auto gridY = rewriter.create<LLVM::TruncOp>(loc, i32Type,
                    rewriter.create<LLVM::SDivOp>(loc,
                    rewriter.create<LLVM::AddOp>(loc, outSize1, cBlockSizeMinusOne), cBlockSize));
    Value gridZ  = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));
    Value blockX = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(blockSize));
    Value blockY = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(blockSize));
    Value blockZ = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));

    SmallVector<Value> params = {
      inBase, inAligned, inOffset, inSize0, inSize1, inStride0, inStride1,
      outBase, outAligned, outOffset, outSize0, outSize1, outStride0, outStride1
    };

    genCuLaunchKernel(rewriter, loc,
                      kernelName, params,
                      gridX, gridY, gridZ,
                      blockX, blockY, blockZ);

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct FullFlowOpConversion : public OpConversionPattern<FullFlowOp> {
  using OpConversionPattern<FullFlowOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(FullFlowOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc   = op.getLoc();
    Value lhs  = adaptor.getLhs();
    Value rhs  = adaptor.getRhs();
    Value bias = adaptor.getBias();
    Value out  = adaptor.getOut();

    auto [lhsBase, lhsAligned, lhsOffset, lhsSize0, lhsSize1, lhsStride0, lhsStride1] = genExtractMemRefMetadata(lhs, rewriter, loc);
    auto [rhsBase, rhsAligned, rhsOffset, rhsSize0, rhsSize1, rhsStride0, rhsStride1] = genExtractMemRefMetadata(rhs, rewriter, loc);
    auto [biasBase, biasAligned, biasOffset, biasSize0, biasSize1, biasStride0, biasStride1] = genExtractMemRefMetadata(bias, rewriter, loc);
    auto [outBase, outAligned, outOffset, outSize0, outSize1, outStride0, outStride1] = genExtractMemRefMetadata(out, rewriter, loc);

    auto elementType = out.getType().cast<MemRefType>().getElementType();
    StringRef kernelName;
    if (elementType.isF32()) kernelName = op.getKernelNameF32();
    else if (elementType.isF16()) kernelName = op.getKernelNameF16();
    else return rewriter.notifyMatchFailure(op, "unsupported element type");

    constexpr int blockSize = 32;
    auto i64Type = rewriter.getI64Type();
    auto i32Type = rewriter.getI32Type();
    auto cBlockSizeMinusOne = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(blockSize - 1));
    auto cBlockSize = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(blockSize));
    auto gridX = rewriter.create<LLVM::TruncOp>(loc, i32Type,
                    rewriter.create<LLVM::SDivOp>(loc,
                    rewriter.create<LLVM::AddOp>(loc, outSize0, cBlockSizeMinusOne), cBlockSize));
    auto gridY = rewriter.create<LLVM::TruncOp>(loc, i32Type,
                    rewriter.create<LLVM::SDivOp>(loc,
                    rewriter.create<LLVM::AddOp>(loc, outSize1, cBlockSizeMinusOne), cBlockSize));
    Value gridZ  = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));
    Value blockX = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(blockSize));
    Value blockY = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(blockSize));
    Value blockZ = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));

    SmallVector<Value> params = {
      lhsBase, lhsAligned, lhsOffset, lhsSize0, lhsSize1, lhsStride0, lhsStride1,
      rhsBase, rhsAligned, rhsOffset, rhsSize0, rhsSize1, rhsStride0, rhsStride1,
      biasBase, biasAligned, biasOffset, biasSize0, biasSize1, biasStride0, biasStride1,
      outBase, outAligned, outOffset, outSize0, outSize1, outStride0, outStride1
    };

    genCuLaunchKernel(rewriter, loc,
                      kernelName, params,
                      gridX, gridY, gridZ,
                      blockX, blockY, blockZ);

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

struct FCGPassNVVMCPU : public impl::FCGPassNVVMCPUBase<FCGPassNVVMCPU> {
  using impl::FCGPassNVVMCPUBase<FCGPassNVVMCPU>::FCGPassNVVMCPUBase;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp topModule = getOperation();
    if (needSkip(topModule)) {
      return;
    }

    TypeConverter converter;
    converter.addConversion(convertTensorToMemRef);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<NVVM::NVVMDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalOp<ModuleOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp func) {
      auto fnType = func.getFunctionType();
      auto hasTensor = [](Type t) { return t.isa<RankedTensorType>(); };
      return !llvm::any_of(fnType.getInputs(), hasTensor) &&
             !llvm::any_of(fnType.getResults(), hasTensor);
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp ret) {
      for (auto operand : ret.getOperands()) {
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

    if (failed(applyFullConversion(topModule, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

private:
  bool needSkip(ModuleOp &topModule) {
    if (getOperation()->getParentOfType<ModuleOp>()) {
      return true;
    }

    StringRef topModuleAttr = "cinder.fcg";
    topModule = getOperation();
    if (topModule->hasAttr(topModuleAttr)) {
      return true;
    }

    topModule->setAttr(topModuleAttr, UnitAttr::get(topModule.getContext()));

    llvm::DenseSet<mlir::TypeID> presentTypeIDs;
    topModule.walk([&](Operation *op) {
      presentTypeIDs.insert(op->getRegisteredInfo()->getTypeID());
    });

    static const llvm::DenseMap<mlir::TypeID, std::pair<StringRef, StringRef>> kernelNameMap = {
        {TypeID::get<MatMulOp>(),   {MatMulOp::getKernelNameF32(),   MatMulOp::getKernelNameF16()}},
        {TypeID::get<AddOp>(),      {AddOp::getKernelNameF32(),      AddOp::getKernelNameF16()}},
        {TypeID::get<ReluOp>(),     {ReluOp::getKernelNameF32(),     ReluOp::getKernelNameF16()}},
        {TypeID::get<FullFlowOp>(), {FullFlowOp::getKernelNameF32(), FullFlowOp::getKernelNameF16()}},
    };

    SmallVector<StringRef, 8> kernelNames;

    for (auto typeID : presentTypeIDs) {
        auto it = kernelNameMap.find(typeID);
        if (it != kernelNameMap.end()) {
            kernelNames.push_back(it->second.first);
            kernelNames.push_back(it->second.second);
        }
    }

    genCUDARuntime(topModule, kernelPath, kernelNames);
    return false;
  }
};

std::unique_ptr<mlir::Pass> createFCGPassNVVMCPU() {
  return std::make_unique<FCGPassNVVMCPU>();
}

} // namespace cinder
