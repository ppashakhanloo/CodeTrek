import python

predicate isInBounds(Scope scope) {
  exists(ExceptStmt e | e.getType().toString() = "HoleException"
  and (e.getScope().getEnclosingScope().containsInScope(scope)
    or e.getScope().getEnclosingScope() = scope))
}