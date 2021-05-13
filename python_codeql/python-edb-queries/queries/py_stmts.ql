import python
import pruning

from Stmt id,int kind,StmtList parent,int idx
where py_stmts(id,kind,parent,idx)
  and isSourceLocation(id.getLocation())
select id,kind,parent,idx
