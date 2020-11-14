// Children of extension method calls are now numbered like other
// method calls

class Expr extends @expr {
  string toString() { none() }
}

class TypeOrTypeRef extends @type_or_ref {
  string toString() { none() }
}

class ExprOrStmtParent extends @exprorstmt_parent  {
  string toString() { none() }
}

class Method extends @method {
  string toString() { none() }
}

predicate isExtensionMethodCall(Expr e) {
  exists(Method m |
    e instanceof @method_invocation_expr and
    expr_call(e, m) and
    params(_, _, _, 0, 4, m, _)
  )
}

from Expr e, int kind, TypeOrTypeRef type, int oldIndex, int newIndex, ExprOrStmtParent parent
where expressions(e, kind, type, oldIndex, parent)
  and (if isExtensionMethodCall(parent) then newIndex = oldIndex - 1 else newIndex = oldIndex)
select e, kind, type, newIndex, parent
