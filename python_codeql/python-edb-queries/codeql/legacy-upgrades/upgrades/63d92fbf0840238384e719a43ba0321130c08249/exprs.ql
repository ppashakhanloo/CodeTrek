class ASTNode extends @ast_node {
  string toString() { none() }
}

class ExprParent extends ASTNode, @exprparent {}

class Expr extends ASTNode, @expr {}

class FunctionExpr extends @functionexpr, Expr {}

class CallSignature extends @call_signature, ASTNode {}

class FunctionTypeExpr extends @functiontypeexpr, ASTNode {}

class FunctionTypeCallSignature extends CallSignature {
  FunctionTypeExpr functionType;

  FunctionTypeCallSignature() {
    properties(this, functionType, _, _, _)
  }

  FunctionTypeExpr getFunctionType() {
    result = functionType
  }
}

from Expr expr, int kind, ExprParent oldParent, int oldIndex, string tostring, int n, int newIndex, ASTNode newParent
where
  exprs(expr, kind, oldParent, oldIndex, tostring) and

  /*
    Parameter default values:
      Before: -4, -6, -8, ...    (start: -4, step: -2)
      After:  -4, -7, -10, ...   (start: -4, step: -3)
  */
  if (oldParent instanceof @function and oldIndex <= -4 and oldIndex % 2 = 0)
  then (n = (oldIndex + 4) / -2 and newIndex = -3 * n - 4 and newParent = oldParent)

  /*
    Class decorators:
      Before: -2, -4, -6, ...    (start: -2, step: -2)
      After:  -2, -5, -8, ...    (start: -2, step: -3)
  */
  else if (oldParent instanceof @classdeclstmt and oldIndex <= -2 and oldIndex % 2 = 0)
  then (n = (oldIndex + 2) / -2 and newIndex = -3 * n - 2 and newParent = oldParent)

  /*
    Function expressions in a function type call signature should be moved up
    to the function type itself (the intermediate call signature is being removed).
  */
  else if (expr instanceof FunctionExpr and oldParent instanceof FunctionTypeCallSignature)
  then (newParent = oldParent.(FunctionTypeCallSignature).getFunctionType() and n = 0 and newIndex = 0)

  // Otherwise retain old index
  else (n = 0 and newIndex = oldIndex and newParent = oldParent)

select expr, kind, newParent, newIndex, tostring
