import java

from Expr id, int kind, Type type, ExprParent parent, int idx
where exprs(id, kind, type, parent, idx)
select id, kind, type, parent, idx