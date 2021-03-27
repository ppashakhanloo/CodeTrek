import python

from Object flow,Scope scope,int kind
where py_scope_flow(flow,scope,kind)
  and scope.inSource()
select flow,scope,kind