import python

from ExprContext id,int kind
where py_expr_contexts(id,kind,id.getParent())
  and id.getParent().(Expr).getScope().inSource()
select id,kind,id.getParent()