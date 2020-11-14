class Expr extends @expr {
  string toString() { none() }
}
class ExprParent extends @exprparent {
  string toString() { none() }
}

from Expr expr, int kind, ExprParent parent, int oldIndex, string tostring, int newIndex, int n, int r
where exprs(expr, kind, parent, oldIndex, tostring)
  and if (parent instanceof @function and oldIndex <= -5)
      then (n = (oldIndex + 5) / 3 and r = (oldIndex + 5) % 3 and newIndex = 4 * n + r - 5)
      else (n = 0 and r = 0 and newIndex = oldIndex)
select expr, kind, parent, newIndex, tostring
