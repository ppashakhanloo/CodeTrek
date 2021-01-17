import java
import pruning

from Class id, string nodeName, Package parentid, Class sourceid
where classes(id, nodeName, parentid, sourceid)
  and isSourceLocation(id.getLocation())
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), parentid, sourceid