class Type extends @type {
  string toString() {
    result = "Type"
  }
}

class Derivation extends @derivation {
  string toString() {
    result = "Derivation"
  }
}

class Specifier extends @specifier {
  string toString() {
    result = "Specifier"
  }
}

class Function extends @function {
  string toString() {
    result = "Function"
  }
}

predicate hasVirtualFunction(Type type) {
  exists(Function func |
    member(type, _, func) and
    exists(Specifier spec |
      funspecifiers(func, spec) and
      specifiers(spec, "virtual")
    )
  ) or
  exists(Type baseType, Type baseClass |
    derivations(_, type, _, baseType, _) and
    unspecifiedtype(baseType, baseClass) and
    hasVirtualFunction(baseClass)
  )
}

// Computing the correct offset for every base class is prohibitively difficult,
// so we'll handle only the common case of single inheritance. In all ABIs that
// we support, a derived class 'D' with a single direct base class 'B' will have
// 'B' at offset zero, provided that 'B' is not a virtual base class, and that
// 'D' does not declare any virtual functions unless 'B' also declares at least
// one virtual function.
from Derivation der, Type sub, Type sup, Type supClass
where
  derivations(der, sub, 0, sup, _) and  // The first derivation for 'sub'
  not derivations(_, sub, 1, _, _) and  // No other derivations for 'sub'
  not exists(Specifier spec |
    derspecifiers(der, spec) and
    specifiers(spec, "virtual")
  ) and  // Not a virtual base class
  unspecifiedtype(sup, supClass) and  // Strip typedefs from base type
  (
    // Derived class shares the base class' vtable pointer
    hasVirtualFunction(supClass) or not hasVirtualFunction(sub)
  )
select der, 0
