class Field extends @membervariable {
  Field() {
    fieldoffsets(this, _, _)
  }

  string getName() { membervariables(this, _, result) }
  string toString() { result = getName() }

  pragma[nomagic]
  int getInitializationOrder() {
    exists(Class cls, int memberIndex |
      this = cls.getField(memberIndex) and
      memberIndex = rank[result + 1](int index |
        cls.getField(index).isInitializable()
      )
    )
  }

  predicate isInitializable() {
    any()
  }
}

class BitField extends Field {
  BitField() { bitfield(this,_,_) }

  predicate isAnonymous() {
    getName() = "(unnamed bitfield)"
  }

  override predicate isInitializable() {
    not isAnonymous()
  }
}

class Class extends @usertype {
  Class() {
    (usertypes(this,_,1) or usertypes(this,_,2) or usertypes(this,_,3) or usertypes(this,_,6))
  }

  string toString() { usertypes(this, result, _) }

  Field getField(int index) {
    member(this, index, result)
  }

  Field getAField() {
    result = getField(_)
  }
}

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

class ClassAggregateLiteral extends @aggregateliteral {
  ClassAggregateLiteral() {
    exists (Class c | expr_types(this, c, _))
  }

  Class getType() {
    expr_types(this, result, _)
  }

  Expr getChild(int n) {
    exprparents(result, n, this)
  }

  Expr getFieldExpr(Field field) {
    field = getType().getAField() and
    result = getChild(field.getInitializationOrder())
  }

  string toString() {
    result = "agg. lit. for " + getType().toString()
    }
}


from ClassAggregateLiteral cal, Expr ex, Field field
where ex = cal.getFieldExpr(field)
select cal, ex, field
