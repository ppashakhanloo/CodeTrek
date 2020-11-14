
class ConditionalExpr extends @conditionalexpr { string toString() { none() } }
class Expr extends @expr { string toString() { none() } }

from ConditionalExpr ce, Expr e
where exprparents(e, 0, ce)
  and exprparents(e, 1, ce)
select ce

