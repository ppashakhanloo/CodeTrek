
class Stmt   extends @stmt                { string toString() { none() } }
class StmtDo extends @stmt_end_test_while { string toString() { none() } }

from StmtDo d, Stmt s
where stmtparents(s, 1, d)
select d, s

