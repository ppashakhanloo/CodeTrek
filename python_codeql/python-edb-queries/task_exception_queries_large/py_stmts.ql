import python
import restrict_boundaries

from Stmt id,int kind,StmtList parent,int idx
where py_stmts(id,kind,parent,idx)
  and id.getScope().inSource()
  and isInBounds(id.getScope())
  and isInBounds(parent.getAnItem().getScope())
select id,kind,parent,idx