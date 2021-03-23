import python

from Cmpop id,int kind,CmpopList parent,int idx
where py_cmpops(id,kind,parent,idx)
  and parent.getParent().getScope().inSource()
select id,kind,parent,idx