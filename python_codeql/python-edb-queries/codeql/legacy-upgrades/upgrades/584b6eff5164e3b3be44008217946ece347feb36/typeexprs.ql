class TypeExpr extends @typeexpr {
  string toString() { none() }
}
class TypeExprParent extends @typeexpr_parent {
  string toString() { none() }
}

// Index -4 is now reserved for the this parameter type.
// Shift everything at that index or below down by one.

from TypeExpr expr, int kind, TypeExprParent parent, int oldIndex, string tostring, int newIndex
where typeexprs(expr, kind, parent, oldIndex, tostring)
  and if (parent instanceof @function and oldIndex <= -4)
      then (newIndex = oldIndex - 1)
      else (newIndex = oldIndex)
select expr, kind, parent, newIndex, tostring
