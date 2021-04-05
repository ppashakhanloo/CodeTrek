import python

from Boolop id,int kind,BoolExpr parent
where py_boolops(id,kind,parent)
  and parent.getScope().inSource()
select id,kind,parent