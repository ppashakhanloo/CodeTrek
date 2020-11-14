
class ExprParent extends @exprparent {
  string toString() { none() }
}

class Expr extends @expr {
  string toString() { none() }
}

from Expr e, int oldKind, int newKind, ExprParent parent, int idx, string tostring
where exprs(e, oldKind, parent, idx, tostring) and
      (if decl(e, _) then newKind = 78
       else if bind(e, _) then newKind = 79
       else newKind = oldKind)
select e, newKind, parent, idx, tostring