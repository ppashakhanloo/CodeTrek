import python

from Operator id,int kind,BinaryExpr parent
where py_operators(id,kind,parent)
  and parent.getScope().inSource()
select id,kind,parent