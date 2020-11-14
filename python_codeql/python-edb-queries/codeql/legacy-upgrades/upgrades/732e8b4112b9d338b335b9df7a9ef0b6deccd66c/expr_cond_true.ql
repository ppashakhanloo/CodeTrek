
class ConditionalExpr extends @conditionalexpr { string toString() { none() } }
class Expr extends @expr { string toString() { none() } }

from ConditionalExpr ce, Expr e
where if exists(Expr guard | exprparents(guard, 0, ce)
                         and exprparents(guard, 1, ce))
      then exprparents(e, -1, ce)
      else exprparents(e, 1, ce)
select ce, e

