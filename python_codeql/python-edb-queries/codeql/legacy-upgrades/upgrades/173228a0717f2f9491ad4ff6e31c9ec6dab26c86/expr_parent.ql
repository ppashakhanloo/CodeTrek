class Expr extends @expr { string toString() { none() } }
class ExprOrStmtParent extends @exprorstmt_parent { string toString() { none() } }

from Expr e, int index, ExprOrStmtParent parent
where expressions(e, _, _, index, parent)
  and (parent instanceof @expr or parent instanceof @stmt)
select e, index, parent
