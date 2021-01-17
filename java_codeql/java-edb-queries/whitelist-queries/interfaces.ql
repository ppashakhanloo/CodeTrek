import java
import pruning

from Interface id, string nodeName, Package parentid, Interface sourceid
where interfaces(id, nodeName, parentid, sourceid)
  and isSourceLocation(id.getLocation())
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), parentid, sourceid