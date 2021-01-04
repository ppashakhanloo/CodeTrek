import python
import pruning

from Location id,Scope scope
where py_scope_location(id,scope)
  and isSourceLocation(id)
select id,scope