class Function extends @function { string toString() { none() } }
class Stmt extends @stmt { string toString() { none() } }
class Block extends @stmt_block { string toString() { none() } }
class TryStmt extends @stmt_try_block { string toString() { none() } }

class FunctionTryStmt extends TryStmt {
  FunctionTryStmt() {
    not stmtparents(this, _, _)
  }
}

from Function f, Stmt s
where if exists(FunctionTryStmt fts | stmtfunction(fts,f))
      then stmtfunction(s.(FunctionTryStmt), f)
      else blockscope(s.(Block), f)
select f, s
