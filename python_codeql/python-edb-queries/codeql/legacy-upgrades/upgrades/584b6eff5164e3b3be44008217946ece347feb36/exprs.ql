class Expr extends @expr {
  string toString() { none() }
}
class ExprParent extends @exprparent {
  string toString() { none() }
}

// Index -4 is now reserved for the this parameter type.
// Shift everything at that index or below down by one.

from Expr expr, int kind, ExprParent parent, int oldIndex, string tostring, int newIndex
where exprs(expr, kind, parent, oldIndex, tostring)
  and if (parent instanceof @function and oldIndex <= -4)
      then (newIndex = oldIndex - 1)
      else (newIndex = oldIndex)
select expr, kind, parent, newIndex, tostring
