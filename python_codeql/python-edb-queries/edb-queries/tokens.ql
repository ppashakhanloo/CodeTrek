import python
import external.CodeDuplication

from Copy id,int offset,int beginLine,int beginColumn,int endLine,int endColumn
where tokens(id,offset,beginLine,beginColumn,endLine,endColumn)
select id,offset,beginLine,beginColumn,endLine,endColumn