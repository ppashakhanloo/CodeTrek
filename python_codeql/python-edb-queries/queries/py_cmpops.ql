import python
import pruning

from Cmpop id,int kind,CmpopList parent,int idx
where py_cmpops(id,kind,parent,idx)
  and isSourceLocation(parent.getParent().getLocation())
select id,kind,parent,idx
