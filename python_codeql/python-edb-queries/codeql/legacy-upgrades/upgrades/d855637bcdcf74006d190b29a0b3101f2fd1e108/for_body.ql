
class Stmt    extends @stmt     { string toString() { none() } }
class StmtFor extends @stmt_for { string toString() { none() } }

from StmtFor f, Stmt b
where stmtparents(b, 3, f)
select f, b

