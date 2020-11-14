class Property extends @property {
  string toString() { none() }
}

from Property prop
where isAbstractMember(prop)
  and not exists (@functiontypeexpr fun | properties(prop, fun, _, _, _))
select prop
