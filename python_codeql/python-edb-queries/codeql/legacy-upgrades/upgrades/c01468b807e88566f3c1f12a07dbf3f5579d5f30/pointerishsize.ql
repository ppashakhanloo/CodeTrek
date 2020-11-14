class DerivedType extends @derivedtype {
  string toString() { result = "" }
}

from DerivedType t, int size
where pointerishsize(t, size)
select t, size, size  // Assume alignment is the same as size.
