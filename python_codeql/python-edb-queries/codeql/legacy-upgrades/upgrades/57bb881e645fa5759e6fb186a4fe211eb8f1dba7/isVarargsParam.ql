class Parameter extends @param {
  string toString() { none() }
}

class Caller extends @caller {
  string toString() { none() }
}

class Callable extends @callable {
  string toString() { none() }
}


from Callable callable, int i, Parameter parm, Caller call, int k
where
  // parm is the ith parameter of callable
  params(parm, _, _, i, callable, _) and

  // the callable has i parameters
  i = max(int j | params(_, _, _, j, callable, _)) and

  // call is a call to the callable
  callableBinding(call, callable) and

  // the call has more than i arguments
  exprs(_, _, _, call, k) and
  k > i
select parm
