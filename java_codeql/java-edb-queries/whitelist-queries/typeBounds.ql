import java
import pruning

from TypeBound id, Type typeid, int pos, BoundedType parentid
where typeBounds(id, typeid, pos, parentid)
  and isSourceLocation(parentid.getLocation())
select id, typeid, pos, parentid