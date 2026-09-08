#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"

#include "cinder/Dialect/FCG/FCG.h"

using namespace mlir;
using namespace cinder;

LogicalResult MatMulOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto outType = getOut().getType();
  auto resultType = getResult().getType();

  auto getShapeAndElemType = [](Type type) -> std::optional<std::pair<ArrayRef<int64_t>, Type>> {
    if (auto tensorType = type.dyn_cast<RankedTensorType>())
      return std::make_pair(tensorType.getShape(), tensorType.getElementType());
    return std::nullopt;
  };

  auto lhsInfo = getShapeAndElemType(lhsType);
  auto rhsInfo = getShapeAndElemType(rhsType);
  auto outInfo = getShapeAndElemType(outType);
  auto resultInfo = getShapeAndElemType(resultType);
  if (!lhsInfo || !rhsInfo || !outInfo || !resultInfo)
    return emitOpError("requires ranked tensor types");

  auto [lhsShape, lhsElem] = *lhsInfo;
  auto [rhsShape, rhsElem] = *rhsInfo;
  auto [outShape, outElem] = *outInfo;
  auto [resultShape, resultElem] = *resultInfo;

  if (!lhsElem.isF32() && !lhsElem.isF16())
    return emitOpError("element type must be F32 or F16, got ") << lhsElem;
  if (lhsElem != rhsElem || lhsElem != outElem || lhsElem != resultElem)
    return emitOpError("element types must match across operands and result");

  if (outShape != resultShape)
    return emitOpError("out shape must exactly match result shape");

  if (lhsShape.size() != 2 || rhsShape.size() != 2 || resultShape.size() != 2)
    return emitOpError("all operands must be 2D tensors");

  auto isCompatible = [](int64_t a, int64_t b) {
    return a == ShapedType::kDynamic || b == ShapedType::kDynamic || a == b;
  };

  if (!isCompatible(lhsShape[1], rhsShape[0]))
    return emitOpError("lhs cols (") << lhsShape[1] << ") vs rhs rows (" << rhsShape[0] << ") incompatible";

  if (!isCompatible(resultShape[0], lhsShape[0]))
    return emitOpError("result rows (") << resultShape[0] << ") vs lhs rows (" << lhsShape[0] << ") incompatible";

  if (!isCompatible(resultShape[1], rhsShape[1]))
    return emitOpError("result cols (") << resultShape[1] << ") vs rhs cols (" << rhsShape[1] << ") incompatible";

  return success();
}

LogicalResult AddOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto outType = getOut().getType();
  auto resultType = getResult().getType();

  auto getShapeAndElemType = [](Type type) -> std::optional<std::pair<ArrayRef<int64_t>, Type>> {
    if (auto tensorType = type.dyn_cast<RankedTensorType>())
      return std::make_pair(tensorType.getShape(), tensorType.getElementType());
    return std::nullopt;
  };

  auto lhsInfo = getShapeAndElemType(lhsType);
  auto rhsInfo = getShapeAndElemType(rhsType);
  auto outInfo = getShapeAndElemType(outType);
  auto resultInfo = getShapeAndElemType(resultType);
  if (!lhsInfo || !rhsInfo || !outInfo || !resultInfo)
    return emitOpError("requires ranked tensor types");

  auto [lhsShape, lhsElem] = *lhsInfo;
  auto [rhsShape, rhsElem] = *rhsInfo;
  auto [outShape, outElem] = *outInfo;
  auto [resultShape, resultElem] = *resultInfo;

  if (!lhsElem.isF32() && !lhsElem.isF16())
    return emitOpError("element type must be F32 or F16, got ") << lhsElem;
  if (lhsElem != rhsElem || lhsElem != outElem || lhsElem != resultElem)
    return emitOpError("element types must match across operands and result");

  if (outShape != resultShape)
    return emitOpError("out shape must exactly match result shape");

  if (lhsShape.size() != 2 || rhsShape.size() != 2 || resultShape.size() != 2)
    return emitOpError("all operands must be 2D tensors");
  if (lhsShape != rhsShape || lhsShape != resultShape)
    return emitOpError("all operands must have identical shapes");

  return success();
}

LogicalResult ReluOp::verify() {
  auto inputType = getInput().getType();
  auto outType = getOut().getType();
  auto resultType = getResult().getType();

  auto getShapeAndElemType = [](Type type) -> std::optional<std::pair<ArrayRef<int64_t>, Type>> {
    if (auto tensorType = type.dyn_cast<RankedTensorType>())
      return std::make_pair(tensorType.getShape(), tensorType.getElementType());
    return std::nullopt;
  };

  auto inputInfo = getShapeAndElemType(inputType);
  auto outInfo = getShapeAndElemType(outType);
  auto resultInfo = getShapeAndElemType(resultType);
  if (!inputInfo || !outInfo || !resultInfo)
    return emitOpError("requires ranked tensor types");

  auto [inputShape, inputElem] = *inputInfo;
  auto [outShape, outElem] = *outInfo;
  auto [resultShape, resultElem] = *resultInfo;

  if (!inputElem.isF32() && !inputElem.isF16())
    return emitOpError("element type must be F32 or F16, got ") << inputElem;
  if (inputElem != outElem || inputElem != resultElem)
    return emitOpError("element types must match across operands and result");

  if (outShape != resultShape)
    return emitOpError("out shape must exactly match result shape");

  if (inputShape.size() != 2 || resultShape.size() != 2)
    return emitOpError("all operands must be 2D tensors");
  if (inputShape != resultShape)
    return emitOpError("input and result shapes must be identical");

  return success();
}

LogicalResult FullFlowOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto biasType = getBias().getType();
  auto outType = getOut().getType();
  auto resultType = getResult().getType();

  auto getShapeAndElemType = [](Type type) -> std::optional<std::pair<ArrayRef<int64_t>, Type>> {
    if (auto tensorType = type.dyn_cast<RankedTensorType>())
      return std::make_pair(tensorType.getShape(), tensorType.getElementType());
    return std::nullopt;
  };

  auto lhsInfo = getShapeAndElemType(lhsType);
  auto rhsInfo = getShapeAndElemType(rhsType);
  auto biasInfo = getShapeAndElemType(biasType);
  auto outInfo = getShapeAndElemType(outType);
  auto resultInfo = getShapeAndElemType(resultType);
  if (!lhsInfo || !rhsInfo || !biasInfo || !outInfo || !resultInfo)
    return emitOpError("requires ranked tensor types");

  auto [lhsShape, lhsElem] = *lhsInfo;
  auto [rhsShape, rhsElem] = *rhsInfo;
  auto [biasShape, biasElem] = *biasInfo;
  auto [outShape, outElem] = *outInfo;
  auto [resultShape, resultElem] = *resultInfo;

  if (!lhsElem.isF32() && !lhsElem.isF16())
    return emitOpError("element type must be F32 or F16, got ") << lhsElem;
  if (lhsElem != rhsElem || lhsElem != biasElem || lhsElem != outElem || lhsElem != resultElem)
    return emitOpError("element types must match across all operands and result");

  if (outShape != resultShape)
    return emitOpError("out shape must exactly match result shape");

  if (biasShape != resultShape)
    return emitOpError("bias shape must exactly match result shape");

  if (lhsShape.size() != 2 || rhsShape.size() != 2 || biasShape.size() != 2 ||
      resultShape.size() != 2 || outShape.size() != 2)
    return emitOpError("all operands must be 2D tensors");

  auto isCompatible = [](int64_t a, int64_t b) {
    return a == ShapedType::kDynamic || b == ShapedType::kDynamic || a == b;
  };

  if (!isCompatible(lhsShape[1], rhsShape[0]))
    return emitOpError("lhs cols (") << lhsShape[1] << ") vs rhs rows (" << rhsShape[0] << ") incompatible";

  if (!isCompatible(resultShape[0], lhsShape[0]))
    return emitOpError("result rows (") << resultShape[0] << ") vs lhs rows (" << lhsShape[0] << ") incompatible";

  if (!isCompatible(resultShape[1], rhsShape[1]))
    return emitOpError("result cols (") << resultShape[1] << ") vs rhs cols (" << rhsShape[1] << ") incompatible";

  return success();
}
