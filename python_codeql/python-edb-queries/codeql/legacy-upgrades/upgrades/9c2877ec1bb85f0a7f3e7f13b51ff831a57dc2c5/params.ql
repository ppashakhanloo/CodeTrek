class Parameter extends @param {
  string toString() { result = "parameter" }
}

class Type extends @type {
  string toString() { result = "type" }
}

class Callable extends @callable {
  string toString() { result = "callable" }
}

from Parameter p, Type t, int pos, Callable c, Parameter src
where
  params(p, _, t, pos, c, src)
select p, t, pos, c, src
