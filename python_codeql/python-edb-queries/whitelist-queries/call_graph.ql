import python

from FunctionInvocation fi, Function f1, Function f2
where not fi.getFunction().isBuiltin() and f1.getFunctionObject() = fi.getCaller().getFunction() and f2.getFunctionObject() = fi.getFunction()
select f1.getDefinition(), f2.getDefinition()
