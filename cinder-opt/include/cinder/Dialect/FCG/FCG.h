#ifndef __FCG_H__
#define __FCG_H__

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringRef.h"

using llvm::StringRef;

#include "cinder/Dialect/FCG/FCGDialect.h.inc"

#define GET_OP_CLASSES
#include "cinder/Dialect/FCG/FCG.h.inc"

#endif // __FCG_H__
