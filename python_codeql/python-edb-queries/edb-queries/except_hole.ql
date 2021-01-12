import python

from ExceptStmt s, int kind, StmtList parent, int idx
where py_stmts(s, kind, parent, idx)
  and exists (Name exp | exp.getId() = "HoleException" |
                exp = s.getType() or exp = s.getType().getAChildNode+())
select s, kind, parent, idx
