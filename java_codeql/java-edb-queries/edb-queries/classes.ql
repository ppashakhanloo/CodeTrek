import java

from Class id, string nodeName, Package parentid, Class sourceid
where classes(id, nodeName, parentid, sourceid)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), parentid, sourceid