class Property extends @property {
  string toString() { none() }
}

class PropertyParent extends @property_parent {
  string toString() { none() }
}

from Property prop, PropertyParent parent, int index, int kind, string tostring
where properties(prop, parent, index, kind, tostring)
  and not parent instanceof @functiontypeexpr
select prop, parent, index, kind, tostring
