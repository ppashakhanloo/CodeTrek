import java
import extra_classes

from string name, string value, MyNamedExprOrStmt parent
where namestrings(name, value, parent)
select name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       value.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       parent