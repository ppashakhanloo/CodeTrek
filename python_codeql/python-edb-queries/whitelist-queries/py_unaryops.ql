import python
import pruning

from Unaryop id,int kind,UnaryExpr parent
where py_unaryops(id,kind,parent)
  and isSourceLocation(parent.getLocation())
select id,kind,parent