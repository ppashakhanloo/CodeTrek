import cpp

from Location id,Container container,int startLine,int startColumn,int endLine,int endColumn
where locations_expr(id,container,startLine,startColumn,endLine,endColumn)
select id,container,startLine,startColumn,endLine,endColumn