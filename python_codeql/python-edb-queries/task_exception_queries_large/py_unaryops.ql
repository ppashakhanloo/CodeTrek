import python

from Unaryop id,int kind,UnaryExpr parent
where py_unaryops(id,kind,parent)
  and parent.getScope().inSource()
select id,kind,parent