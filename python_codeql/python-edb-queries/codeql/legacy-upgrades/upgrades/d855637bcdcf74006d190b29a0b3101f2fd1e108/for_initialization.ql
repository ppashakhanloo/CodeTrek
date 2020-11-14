
class Stmt    extends @stmt     { string toString() { none() } }
class StmtFor extends @stmt_for { string toString() { none() } }

from StmtFor f, Stmt s
where stmtparents(s, 0, f)
select f, s

