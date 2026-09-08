#ifndef _FCGDIALECT_TD__
#define _FCGDIALECT_TD__

include "mlir/IR/DialectBase.td"

def FCGDialect : Dialect {
  let name = "fcg";
  let cppNamespace = "::cinder";

  let summary = "Fast Computation Graph Dialect";
  let description = [{
    A high-level, dynamically-shaped tensor dialect for AI inference graphs.
  }];
}

#endif // _FCGDIALECT_TD__
