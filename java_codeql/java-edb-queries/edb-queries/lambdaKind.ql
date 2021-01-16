import java

from LambdaExpr exprId, int bodyKind
where lambdaKind(exprId, bodyKind)
select exprId, bodyKind