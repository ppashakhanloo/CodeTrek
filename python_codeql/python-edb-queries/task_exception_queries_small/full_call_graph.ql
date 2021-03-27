import python

from FunctionInvocation fi, Function f1, Function f2
where not fi.getFunction().isBuiltin()
  and f1.getFunctionObject() = fi.getCaller().getFunction()
  and f2.getFunctionObject() = fi.getFunction()
  and f1.getScope().inSource()
  and f2.getScope().inSource()
select f1.getDefinition(), f2.getDefinition(), f1, f2, fi.getCall()
