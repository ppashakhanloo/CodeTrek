import java
import pruning

from Annotation parentid, Method id2, Expr value
where annotValue(parentid, id2, value)
  and isSourceLocation(id2.getLocation())
select parentid, id2, value