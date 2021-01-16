import java
import extra_classes

from Import id, MyTypeOrPackage holder, string name, int kind
where imports(id, holder, name, kind)
select id, holder, name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"), kind