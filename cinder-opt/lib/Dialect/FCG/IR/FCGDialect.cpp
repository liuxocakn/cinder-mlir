#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "cinder/Dialect/FCG/FCG.h"
#include "cinder/Dialect/FCG/FCGDialect.cpp.inc"

#define GET_OP_CLASSES
#include "cinder/Dialect/FCG/FCG.cpp.inc"

using namespace mlir;
using namespace cinder;

void FCGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cinder/Dialect/FCG/FCG.cpp.inc"
  >();
}
