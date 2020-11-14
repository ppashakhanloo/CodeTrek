class TypeExpr extends @typeexpr {
  string toString() { none() }
}
class TypeExprParent extends @typeexpr_parent {
  string toString() { none() }
}

from TypeExpr expr, int kind, TypeExprParent parent, int oldIndex, string tostring, int newIndex, int n, int r
where typeexprs(expr, kind, parent, oldIndex, tostring)
  and if (parent instanceof @function and oldIndex <= -5)
      then (n = (oldIndex + 5) / 3 and r = (oldIndex + 5) % 3 and newIndex = 4 * n + r - 5)
      else (n = 0 and r = 0 and newIndex = oldIndex)
select expr, kind, parent, newIndex, tostring
