import java
import extra_classes
import pruning

from Wildcard id, string nodeName, int kind
where wildcards(id, nodeName, kind)
  and isSourceLocation(id.getLocation())
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), kind