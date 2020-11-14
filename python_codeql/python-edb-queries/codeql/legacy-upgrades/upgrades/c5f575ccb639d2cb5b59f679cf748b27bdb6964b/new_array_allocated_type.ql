
class NewArrayExpr extends @new_array_expr { string toString() { none() } }
class Type extends @type { string toString() { none() } }

from NewArrayExpr e, Type t
where expr_types(e, t)
select e, t

