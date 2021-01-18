import java

from Modifier id, string nodeName
where modifiers(id, nodeName)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")