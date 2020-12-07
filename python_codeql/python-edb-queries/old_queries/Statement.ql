import python

from Stmt id, int kind, int idx
where py_stmts(id, kind, _, idx)
select id, kind, id.getParent(), idx
