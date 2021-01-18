import java
import pruning

from Expr id, Callable callable_id
where callableEnclosingExpr(id, callable_id)
  and isSourceLocation(id.getLocation())
select id, callable_id