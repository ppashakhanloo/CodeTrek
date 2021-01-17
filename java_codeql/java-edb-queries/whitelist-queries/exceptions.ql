import java
import pruning

from Exception id, Type typeid, Callable parentid
where exceptions(id, typeid, parentid)
  and isSourceLocation(parentid.getLocation())
select id, typeid, parentid