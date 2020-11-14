// Change the index of class decorators to make room for the implements clause.

// Old index: -1, -2, -3, ...
// New index: -2, -4, -6, ...

class Expr extends @expr {
  string toString() { none() }
}
class ExprParent extends @exprparent {
  string toString() { none() }
}

from Expr expr, int kind, ExprParent parent, int oldIndex, string tostring, int newIndex
where exprs(expr, kind, parent, oldIndex, tostring)
  and if (parent instanceof @classdeclstmt and oldIndex <= -1)
      then (newIndex = 2 * oldIndex)
      else (newIndex = oldIndex)
select expr, kind, parent, newIndex, tostring
