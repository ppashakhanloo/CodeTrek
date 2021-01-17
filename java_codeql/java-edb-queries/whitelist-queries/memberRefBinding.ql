import java
import pruning

from Expr id, Callable callable
where memberRefBinding(id, callable)
  and isSourceLocation(id.getLocation())
select id, callable