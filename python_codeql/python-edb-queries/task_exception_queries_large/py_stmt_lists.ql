import python
import restrict_boundaries

from StmtList id,int idx
where py_stmt_lists(id,id.getParent(),idx)
  and id.getLastItem().getScope().inSource()
  and isInBounds(id.getLastItem().getScope())
select id,id.getParent(),idx