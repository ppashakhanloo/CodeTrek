import java

from Location id, File file, int beginLine, int beginColumn, int endLine, int endColumn
where locations_default(id, file, beginLine, beginColumn, endLine, endColumn)
select id, file, beginLine, beginColumn, endLine, endColumn