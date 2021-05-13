import python
import pruning

from Boolop id,int kind,BoolExpr parent
where py_boolops(id,kind,parent)
  and isSourceLocation(parent.getLocation())
select id,kind,parent
