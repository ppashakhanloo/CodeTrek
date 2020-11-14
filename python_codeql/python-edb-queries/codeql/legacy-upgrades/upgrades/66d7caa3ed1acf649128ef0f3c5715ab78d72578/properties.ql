// Fields now have the kind @proper_field = 8.
// Previously they were @value_property = 0.

class Property extends @property {
  string toString() { none() }
}

class PropertyParent extends @property_parent {
  string toString() { none() }
}

predicate isField(Property prop, int kind, PropertyParent parent) {
  parent instanceof @classorinterface and
  kind = 0 and
  not isMethod(prop)
}

from Property prop, PropertyParent parent, int index, int oldKind, string tostring, int newKind
where properties(prop, parent, index, oldKind, tostring) and
      (if isField(prop, oldKind, parent)
       then newKind = 8
       else newKind = oldKind)
select prop, parent, index, newKind, tostring
