import java
import pruning

from Parameter id, Type typeid, int pos, Callable parentid, Parameter sourceid
where params(id, typeid, pos, parentid, sourceid)
  and isSourceLocation(id.getLocation())
select id, typeid, pos, parentid, sourceid