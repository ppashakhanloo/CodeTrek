import python
import pruning

from Expr id,int kind,int idx
where py_exprs(id,kind,id.getParent(),idx)
  and isSourceLocation(id.getLocation())
select id,kind,id.getParent(),idx
