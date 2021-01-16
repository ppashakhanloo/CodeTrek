import java

from TypeBound id, Type typeid, int pos, BoundedType parentid
where typeBounds(id, typeid, pos, parentid)
select id, typeid, pos, parentid