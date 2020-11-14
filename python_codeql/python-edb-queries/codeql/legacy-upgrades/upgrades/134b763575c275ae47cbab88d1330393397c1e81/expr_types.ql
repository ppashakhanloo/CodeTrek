class Expr extends @expr {
  string toString() {
    result = "Expr"
  }
}

class Type extends @type {
  string toString() {
    result = "Type"
  }
}

// Since computing the correct value category for all expressions is extremely
// difficult from QL, we populate the `value_category` column of the
// `expr_types` with -1, which means that it will not have any value category
// (i.e. none of the is*ValueCategory predicates will hold).
from Expr expr, Type type
where expr_types(expr, type)
select expr, type, -1
