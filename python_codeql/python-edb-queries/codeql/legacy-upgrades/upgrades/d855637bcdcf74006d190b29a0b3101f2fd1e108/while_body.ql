
class Stmt      extends @stmt       { string toString() { none() } }
class StmtWhile extends @stmt_while { string toString() { none() } }

from StmtWhile w, Stmt s
where stmtparents(s, 1, w)
select w, s

