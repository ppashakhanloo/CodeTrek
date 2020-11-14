class BuiltInType extends @builtintype {
  string toString() { result = "" }
}

from BuiltInType t, string name, int kind, int size, int sign
where builtintypes(t, name, kind, size, sign)
select t, name, kind, size, sign, size // Assume alignment is the same as size.
