class PropertyParent extends @ast_node {
  PropertyParent() {
    this instanceof @property_parent
    or
    this instanceof @classdecl
  }
  string toString() { none() }
}

class Property extends @property {
  string toString() { none() }
}

from Property property, PropertyParent parent, int index, int kind, string tostring, int newIndex, PropertyParent newParent
where properties(property, parent, index, kind, tostring) and
  if parent instanceof @class
  then (classes(parent, newParent, _) and newIndex = index + 2)
  else (newParent = parent and newIndex = index)
select property, newParent, newIndex, kind, tostring
