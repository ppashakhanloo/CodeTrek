import python
import pruning

from FunctionInvocation fi, Function f1, Function f2
where not fi.getFunction().isBuiltin()
  and f1.getFunctionObject() = fi.getCaller().getFunction()
  and f2.getFunctionObject() = fi.getFunction()
  and isSourceLocation(f1.getLocation())
  and isSourceLocation(f2.getLocation())
select f1.getDefinition(), f2.getDefinition(), f1, f2
