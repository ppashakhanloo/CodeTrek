import python

from Stmt id,int kind,StmtList parent,int idx
where py_stmts(id,kind,parent,idx)
  and id.getScope().inSource()
select id,kind,parent,idx