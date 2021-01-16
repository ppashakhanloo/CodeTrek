import java

from Interface id, string nodeName, Package parentid, Interface sourceid
where interfaces(id, nodeName, parentid, sourceid)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), parentid, sourceid