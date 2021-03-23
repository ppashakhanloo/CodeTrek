import python

from Location id,Scope scope
where py_scope_location(id,scope)
  and scope.inSource()
select id,scope