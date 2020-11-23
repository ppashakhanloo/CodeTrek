import python

from Stmt id, int kind, StmtList parent
where py_stmts(id, kind, parent, _)
select id, kind, parent
