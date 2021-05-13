import python
import pruning

from ExceptStmt s, int kind, StmtList parent, int idx
where py_stmts(s, kind, parent, idx)
  and isSourceLocation(s.getLocation())
  and exists (Name exp | exp.getId() = "HoleException" |
                exp = s.getType() or exp = s.getType().getAChildNode+())
select s, kind, parent, idx
