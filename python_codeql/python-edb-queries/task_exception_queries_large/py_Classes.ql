import python
import restrict_boundaries

from Class id,ClassExpr parent
where py_Classes(id,parent)
  and id.inSource()
  and isInBounds(id)
select id,parent