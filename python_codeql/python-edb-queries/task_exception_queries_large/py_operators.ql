import python
import restrict_boundaries

from Operator id,int kind,BinaryExpr parent
where py_operators(id,kind,parent)
  and parent.getScope().inSource()
  and isInBounds(parent.getScope())
select id,kind,parent