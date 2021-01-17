import java
import pruning

from Parameter id, string nodeName
where paramName(id, nodeName)
  and isSourceLocation(id.getLocation())
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")