import java
import pruning

from Field id, string nodeName, Type typeid, Type parentid, Field sourceid
where fields(id, nodeName, typeid, parentid, sourceid)
  and isSourceLocation(id.getLocation())
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), typeid, parentid, sourceid