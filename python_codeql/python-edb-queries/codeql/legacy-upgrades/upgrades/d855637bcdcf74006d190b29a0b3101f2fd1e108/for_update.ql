
class Expr    extends @expr     { string toString() { none() } }
class StmtFor extends @stmt_for { string toString() { none() } }

from StmtFor f, Expr e
where exprparents(e, 2, f)
select f, e

