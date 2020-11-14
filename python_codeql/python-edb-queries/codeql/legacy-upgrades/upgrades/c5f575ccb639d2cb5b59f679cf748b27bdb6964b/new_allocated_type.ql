
class NewExpr extends @new_expr { string toString() { none() } }
class Type extends @type { string toString() { none() } }

from NewExpr e, Type t
where expr_types(e, t)
select e, t

