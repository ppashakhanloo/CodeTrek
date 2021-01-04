import python
import pruning

from ExprContext id,int kind
where py_expr_contexts(id,kind,id.getParent())
  and isSourceLocation(id.getParent().(Expr).getLocation())
select id,kind,id.getParent()