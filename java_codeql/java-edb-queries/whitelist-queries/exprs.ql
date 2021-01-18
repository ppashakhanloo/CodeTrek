import java
import pruning

from Expr id, int kind, Type type, ExprParent parent, int idx
where exprs(id, kind, type, parent, idx)
  and isSourceLocation(id.getLocation())
select id, kind, type, parent, idx