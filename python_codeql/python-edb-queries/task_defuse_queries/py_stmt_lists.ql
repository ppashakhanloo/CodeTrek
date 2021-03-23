import python

from StmtList id,int idx
where py_stmt_lists(id,id.getParent(),idx)
  and id.getLastItem().getScope().inSource()
select id,id.getParent(),idx