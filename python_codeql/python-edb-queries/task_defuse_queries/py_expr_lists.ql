import python

from ExprList id,int idx
where py_expr_lists(id,id.getParent(),idx)
  and id.getAnItem().getScope().inSource()
select id,id.getParent(),idx