import python
import pruning

from Class id,ClassExpr parent
where py_Classes(id,parent)
  and isSourceLocation(id.getLocation())
select id,parent
