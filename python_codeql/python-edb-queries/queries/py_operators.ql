import python
import pruning

from Operator id,int kind,BinaryExpr parent
where py_operators(id,kind,parent)
  and isSourceLocation(parent.getLocation())
select id,kind,parent
