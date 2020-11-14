/* library stubs */

class Locatable extends @locatable {
  string getURL() {
    exists (string path, int sl, int sc, int el, int ec |
      hasLocationInfo(path, sl, sc, el, ec) and
      result = "file://" + path + ":" + sl + ":" + sc + ":" + el + ":" + ec
    )
  }
  
  predicate hasLocationInfo(string path, int sl, int sc, int el, int ec) {
    exists (@location loc, @file file |
      hasLocation(this, loc) and
      locations_default(loc, file, sl, sc, el, ec) and
      files(file, path, _, _, _)
    )
  }
  
  string toString() {
    result = getURL()
  }
}

class Element extends Locatable, @element {
  /** child at position old_idx should become child of new_parent at index new_idx */
  predicate move(int old_idx, Element new_parent, int new_idx) {
    // by default nothing changes
    exists (getChild(old_idx)) and
    new_parent = this and
    new_idx = old_idx
  }
  
  Element getChild(int idx) {
    exprs(result, _, _, this, idx) or
    stmts(result, _, this, idx, _)
  }
  
  override string toString() { result = "element" }
}

class Expr extends Element, @expr {
  Element getParent() {
    exprs(this, _ ,_ , result, _)
  }

  override string toString() { result = "expr" }
}

class Stmt extends Element, @stmt {
  /** the index at which the type access should be attached, if any */
  int getTypeAccessIndex() {
    none()
  }
  
  override string toString() { result = "stmt" }
}

class Type extends @type { string toString() { result = "type" } }
class Class extends Type, @class { override string toString() { classes(this, result, _, _) } }
class Primitive extends Type, @primitive { override string toString() { primitives(this, result) } }

class LocalVariableDeclExpr extends Expr, @localvariabledeclexpr {
  override predicate move(int old_idx, Element new_parent, int new_idx) {
    exists (getChild(old_idx)) and
    // type access is moved to parent
    (if old_idx = 0 then
      exists (Stmt parent | parent = getParent() |
        new_parent = parent and new_idx = parent.getTypeAccessIndex()
      )
    // init becomes child 0
    else if old_idx = 1 then
      (new_parent = this and new_idx = 0)
    // everything else stays the same
    else
      (new_parent = this and new_idx = old_idx))
  }
  
  override string toString() { result = "localvariabledeclexpr" }
}

class LocalVariableDeclStmt extends Stmt, @localvariabledeclstmt {
  // LocalVariableDeclExprs get moved from indices 0... to 1...
  override predicate move(int old_idx, Element new_parent, int new_idx) {
    exists (getChild(old_idx)) and
    new_parent = this and
    (if old_idx >= 0 then
      new_idx = old_idx+1
    else
      new_idx = old_idx)
  }
  
  override int getTypeAccessIndex() { result = 0 }
  
  override string toString() { result = "localvariabledeclstmt" }
}

class ForStmt extends Stmt, @forstmt {
  override predicate move(int old_idx, Element new_parent, int new_idx) {
    exists (getChild(old_idx)) and
    new_parent = this and
    // init used to be 0, now is -1; update used to be 2, now is 3; body used to be 3, now is 2
    (if old_idx = 0 then new_idx = -1
    else if old_idx = 2 then new_idx = 3
    else if old_idx = 3 then new_idx = 2
    else new_idx = old_idx)
  }

  override int getTypeAccessIndex() { result = 0 }
  
  override string toString() { result = "forstmt" }
}

class EnhancedForStmt extends Stmt, @enhancedforstmt {
  override int getTypeAccessIndex() { result = -1 }
  
  override string toString() { result = "enhancedforstmt" }
}

class CatchClause extends Stmt, @catchclause {
  override int getTypeAccessIndex() { result = -1 }
  
  override string toString() { result = "catchclause" }
}

class BlockStmt extends Stmt, @block {
  override string toString() { result = "blockstmt" }
}