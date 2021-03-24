import python
import restrict_boundaries

from Cmpop id,int kind,CmpopList parent,int idx
where py_cmpops(id,kind,parent,idx)
  and parent.getParent().getScope().inSource()
  and isInBounds(parent.getParent().getScope())
select id,kind,parent,idx