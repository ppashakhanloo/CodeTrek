import java

from Expr id, Callable callable_id
where callableEnclosingExpr(id, callable_id)
select id, callable_id