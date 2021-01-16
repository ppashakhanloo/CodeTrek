import java
import extra_classes

from TypeVariable id, string nodeName, int pos, int kind, MyTypeOrCallable parentid
where typeVars(id, nodeName, pos, kind, parentid)
select id, nodeName.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       pos, kind, parentid