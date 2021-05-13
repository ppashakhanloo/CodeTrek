import python
import pruning

from StmtList id,int idx
where py_stmt_lists(id,id.getParent(),idx)
  and isSourceLocation(id.getLastItem().getLocation())
select id,id.getParent(),idx
