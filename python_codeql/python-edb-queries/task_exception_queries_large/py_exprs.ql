import python
import restrict_boundaries

from Expr id,int kind,int idx
where py_exprs(id,kind,id.getParent(),idx)
  and id.getScope().inSource()
  and isInBounds(id.getScope())
select id,kind,id.getParent(),idx