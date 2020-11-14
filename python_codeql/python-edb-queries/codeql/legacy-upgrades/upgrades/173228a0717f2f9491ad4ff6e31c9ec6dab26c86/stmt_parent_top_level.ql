class Stmt extends @stmt { string toString() { none() } }
class ExprOrStmtParent extends @exprorstmt_parent { string toString() { none() } }

from Stmt s, int index, ExprOrStmtParent parent
where statements(s, _, index, parent)
  and not parent instanceof @expr
  and not parent instanceof @stmt
select s, index, parent
