import java
import pruning

from Location id, File file, int beginLine, int beginColumn, int endLine, int endColumn
where locations_default(id, file, beginLine, beginColumn, endLine, endColumn)
  and isSourceLocation(id)
select id, file, beginLine, beginColumn, endLine, endColumn