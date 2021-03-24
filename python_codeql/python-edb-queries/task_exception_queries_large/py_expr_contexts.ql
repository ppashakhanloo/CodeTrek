import python
import restrict_boundaries

from ExprContext id,int kind
where py_expr_contexts(id,kind,id.getParent())
  and id.getParent().(Expr).getScope().inSource()
  and isInBounds(id.getParent().(Expr).getScope())
select id,kind,id.getParent()