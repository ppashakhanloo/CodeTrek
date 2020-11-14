class LocationExpr extends @location_expr {
  string toString() {
    exists (int startLine |
      locations_expr(this, _, startLine, _, _, _) and
      result = startLine.toString()
    )
  }
}

class Expr extends @expr {
  string toString() { result = "expr at line " + getLocation().toString() }
  LocationExpr getLocation() { exprs(this, _, result) }
}

class ArrayType extends @derivedtype {
  ArrayType() { derivedtypes(this,_,4,_) }
  string toString() { result = "array type" }
}

class ArrayAggregateLiteral extends @aggregateliteral {
  ArrayAggregateLiteral() {
    exists (ArrayType t | expr_types(this, t, _))
  }

  ArrayType getType() {
    expr_types(this, result, _)
  }

  Expr getChild(int n) {
    exprparents(result, n, this)
  }

  Expr getElementExpr(int elementIndex) {
    result = getChild(elementIndex)
  }

  string toString() {
    result = "agg. lit. for " + getType().toString()
  }
}

from ArrayAggregateLiteral aal, Expr ex, int elementIndex
where ex = aal.getElementExpr(elementIndex)
select aal, ex, elementIndex
