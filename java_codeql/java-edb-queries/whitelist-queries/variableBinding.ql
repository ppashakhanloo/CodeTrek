import java
import pruning

from VarAccess expr, Variable variable
where variableBinding(expr, variable)
  and isSourceLocation(expr.getLocation())
select expr, variable