
class Stmt       extends @stmt        { string toString() { none() } }
class StmtSwitch extends @stmt_switch { string toString() { none() } }

from StmtSwitch s, Stmt b
where stmtparents(b, 1, s)
select s, b

