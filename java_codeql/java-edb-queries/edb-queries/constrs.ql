import java

from Constructor id, string nodeName, string signature, Type typeid, Type parentid, Constructor sourceid
where constrs(id, nodeName, signature, typeid, parentid, sourceid)
select id,
       nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       signature.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       typeid, parentid, sourceid