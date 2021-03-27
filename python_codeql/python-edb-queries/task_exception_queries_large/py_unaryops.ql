import python
import restrict_boundaries

from Unaryop id,int kind,UnaryExpr parent
where py_unaryops(id,kind,parent)
  and parent.getScope().inSource()
  and isInBounds(parent.getScope())
select id,kind,parent