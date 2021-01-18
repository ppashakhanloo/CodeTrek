import java
import extra_classes
import pruning

from Import id, MyTypeOrPackage holder, string name, int kind
where imports(id, holder, name, kind)
  and isSourceLocation(id.getLocation())
select id, holder, name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), kind