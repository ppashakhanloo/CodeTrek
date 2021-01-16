import java

from Variable id, string nodeName, Type typeid, LocalVariableDeclExpr parentid
where localvars(id, nodeName, typeid, parentid)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), typeid, parentid