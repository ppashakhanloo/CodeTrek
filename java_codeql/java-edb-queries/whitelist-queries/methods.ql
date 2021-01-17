import java
import pruning

from Method id, string nodeName, string signature, Type typeid, Type parentid, Method sourceid
where methods(id, nodeName, signature, typeid, parentid, sourceid)
  and isSourceLocation(id.getLocation())
select id,
       nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       signature.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       typeid, parentid, sourceid