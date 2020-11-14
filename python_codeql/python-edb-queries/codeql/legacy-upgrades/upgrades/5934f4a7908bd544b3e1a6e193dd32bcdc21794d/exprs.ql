class Expr extends @expr {
  string toString() { none() }
}

class ExprParent extends @exprparent {
  string toString() { none() }
}

// Default values have a new index to make room for return types and parameter types.

// For the n-th parameter:
//   oldChildIndex = -(n + 3)
//   newChildIndex = -(2n + 4)

// We rewrite the first line to get:
//   n = -oldChildIndex - 3

from Expr expr, int kind, ExprParent parent, int index, string tostring, int newIndex, int n
where exprs(expr, kind, parent, index, tostring)
  and if (parent instanceof @function and index <= -3)
      then (n = -index - 3 and newIndex = -(2 * n + 4))
      else (newIndex = index and n = 0)
select expr, kind, parent, newIndex, tostring
