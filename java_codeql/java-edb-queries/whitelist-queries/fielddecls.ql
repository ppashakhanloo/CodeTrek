import java
import pruning

from FieldDeclaration id, Type parentid
where fielddecls(id, parentid)
  and isSourceLocation(id.getLocation())
select id, parentid