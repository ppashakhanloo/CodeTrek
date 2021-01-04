import python
import pruning

from Object flow,Scope scope,int kind
where py_scope_flow(flow,scope,kind)
  and isSourceLocation(scope.getLocation())
select flow,scope,kind