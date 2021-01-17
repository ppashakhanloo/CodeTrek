import java
import pruning

from LambdaExpr exprId, int bodyKind
where lambdaKind(exprId, bodyKind)
  and isSourceLocation(exprId.getLocation())
select exprId, bodyKind