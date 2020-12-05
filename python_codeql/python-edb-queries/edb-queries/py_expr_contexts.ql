import python

from ExprContext id,int kind
where py_expr_contexts(id,kind,id.getParent())
select id,kind,id.getParent()