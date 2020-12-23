import python

from Expr id, int kind, int idx
where py_exprs(id, kind, _, idx)
select id, kind, id.getParent(), idx
