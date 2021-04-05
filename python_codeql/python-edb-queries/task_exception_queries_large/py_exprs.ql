import python

from Expr id,int kind,int idx
where py_exprs(id,kind,id.getParent(),idx)
  and id.getScope().inSource()
select id,kind,id.getParent(),idx