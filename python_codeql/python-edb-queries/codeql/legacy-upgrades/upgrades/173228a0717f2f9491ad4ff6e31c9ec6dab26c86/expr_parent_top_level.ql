class Expr extends @expr { string toString() { none() } }
class ExprOrStmtParent extends @exprorstmt_parent { string toString() { none() } }

from Expr e, int index, int index_adjusted, ExprOrStmtParent parent
where expressions(e, _, _, index, parent)
  and not parent instanceof @expr
  and not parent instanceof @stmt
  and if parent instanceof @indexer and index = 4 then
    index_adjusted = 0
  else if parent instanceof @property and index = 3 then
    index_adjusted = 1
  else if parent instanceof @property and index = 4 then
    index_adjusted = 0
  else
    index_adjusted = index
select e, index_adjusted, parent
