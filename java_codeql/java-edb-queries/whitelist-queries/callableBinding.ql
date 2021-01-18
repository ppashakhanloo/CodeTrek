import java
import pruning

from Call callerid, Callable callee
where callableBinding(callerid, callee)
  and isSourceLocation(callerid.getLocation())
select callerid, callee