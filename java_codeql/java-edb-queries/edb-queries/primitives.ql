import java

from Type id, string nodeName
where primitives(id, nodeName)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")