import Lib

from Expr e, int kind, Type type, Element old_parent, int old_idx, Element new_parent, int new_idx
where exprs(e, kind, type, old_parent, old_idx) and
      old_parent.move(old_idx, new_parent, new_idx)
select e, kind, type, new_parent, new_idx