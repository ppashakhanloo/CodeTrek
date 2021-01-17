import java
import pruning

from Stmt id, int kind, StmtParent parent, int idx, Callable bodydecl
where stmts(id, kind, parent, idx, bodydecl)
  and isSourceLocation(id.getLocation())
select id, kind, parent, idx, bodydecl