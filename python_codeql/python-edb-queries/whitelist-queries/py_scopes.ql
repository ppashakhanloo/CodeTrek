import python
import extra_classes
import pruning

from ExprOrStmt node,Scope scope
where py_scopes(node,scope)
  and isSourceLocation(scope.getLocation())
select node,scope