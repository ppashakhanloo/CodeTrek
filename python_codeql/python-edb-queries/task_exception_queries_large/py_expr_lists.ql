import python
import restrict_boundaries

from ExprList id,int idx
where py_expr_lists(id,id.getParent(),idx)
  and id.getAnItem().getScope().inSource()
  and isInBounds(id.getAnItem().getScope())
select id,id.getParent(),idx