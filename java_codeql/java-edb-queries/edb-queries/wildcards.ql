import java
import extra_classes

from Wildcard id, string nodeName, int kind
where wildcards(id, nodeName, kind)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), kind