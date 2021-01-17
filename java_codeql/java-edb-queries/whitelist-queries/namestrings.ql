import java
import extra_classes
import pruning

from string name, string value, MyNamedExprOrStmt parent
where namestrings(name, value, parent)
  and isSourceLocation(parent.getLocation())
select name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       value.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       parent