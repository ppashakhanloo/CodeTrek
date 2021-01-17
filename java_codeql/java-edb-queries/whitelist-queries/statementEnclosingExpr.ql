import java
import pruning

from Expr id, Stmt statment_id
where statementEnclosingExpr(id, statment_id)
  and isSourceLocation(id.getLocation())
select id, statment_id