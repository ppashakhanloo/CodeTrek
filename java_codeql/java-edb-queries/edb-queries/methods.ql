import java

from Method id, string nodeName, string signature, Type typeid, Type parentid, Method sourceid
where methods(id, nodeName, signature, typeid, parentid, sourceid)
select id,
       nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       signature.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       typeid, parentid, sourceid