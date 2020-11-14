class TypeExpr extends @typeexpr {
  string toString() { none() }
}

class TypeExprParent extends @typeexpr_parent {
  string toString() { none() }
}

class CallSignature extends @call_signature {
  string toString() { none() }

  predicate isConstructor() { this instanceof @constructor_call_signature }
}

class FunctionTypeExpr extends @functiontypeexpr, TypeExpr {
  CallSignature signature;

  FunctionTypeExpr() {
    properties(signature, this, _, _, _)
  }

  int getNewKind() {
    if signature.isConstructor()
    then result = 24
    else result = 23
  }
}

from TypeExpr type, int oldKind, TypeExprParent parent, int oldIndex, string tostring, int n, int newIndex, int newKind
where
  typeexprs(type, oldKind, parent, oldIndex, tostring) and

  /*
    Class superinterfaces:
      Before: -1, -3, -5, ...    (start: -1, step: -2)
      After:  -1, -4, -7, ...    (start: -1, step: -3)
  */
  if (parent instanceof @classdefinition and oldIndex <= -1 and oldIndex % 2 = -1)
  then (n = (oldIndex + 1) / -2 and newIndex = -3 * n - 1 and newKind = oldKind)

  /*
    Interface superinterfaces:
      Before: -1, -2, -3, ...    (start: -1, step: -1)
      After:  -1, -3, -5, ...    (start: -1, step: -2)
  */
  else if (parent instanceof @interfacedeclaration and oldIndex <= -1)
  then (n = (oldIndex + 1) / -1 and newIndex = -2 * n - 1 and newKind = oldKind)

  /*
    Function type:
      Before: kind = 22, contains a call signature containing the function expression
      After: kind = 23 (plain) or 24 (constructor), directly contains the function expression
  */
  else if (type instanceof FunctionTypeExpr)
  then (newKind = type.(FunctionTypeExpr).getNewKind() and n = 0 and newIndex = oldIndex)

  else (n = 0 and newIndex = oldIndex and newKind = oldKind)

select type, newKind, parent, newIndex, tostring
