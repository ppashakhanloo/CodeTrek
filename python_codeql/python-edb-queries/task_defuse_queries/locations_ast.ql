import python

from Location id,Module module_,int beginLine,int beginColumn,int endLine,int endColumn
where locations_ast(id,module_,beginLine,beginColumn,endLine,endColumn)
  and module_.inSource()
select id,module_,beginLine,beginColumn,endLine,endColumn