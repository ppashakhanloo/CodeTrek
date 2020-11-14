
class Stmt   extends @stmt    { string toString() { none() } }
class StmtIf extends @stmt_if { string toString() { none() } }

from StmtIf i, Stmt s
where stmtparents(s, 2, i)
select i, s

