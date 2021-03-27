import python
import restrict_boundaries

from Boolop id,int kind,BoolExpr parent
where py_boolops(id,kind,parent)
  and parent.getScope().inSource()
  and isInBounds(parent.getScope())
select id,kind,parent