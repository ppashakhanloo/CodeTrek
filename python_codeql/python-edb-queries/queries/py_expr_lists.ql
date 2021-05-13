import python
import pruning

from ExprList id,int idx
where py_expr_lists(id,id.getParent(),idx)
  and isSourceLocation(id.getAnItem().getLocation())
select id,id.getParent(),idx
