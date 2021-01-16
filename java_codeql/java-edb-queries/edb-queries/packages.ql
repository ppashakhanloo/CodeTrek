import java

from Package id, string nodeName
where packages(id, nodeName)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")