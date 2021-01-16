import java

from Parameter id, string nodeName
where paramName(id, nodeName)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")