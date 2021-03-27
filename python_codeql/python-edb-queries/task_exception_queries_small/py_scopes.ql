import python
import extra_classes

from ExprOrStmt node,Scope scope
where py_scopes(node,scope)
  and scope.inSource()
select node,scope