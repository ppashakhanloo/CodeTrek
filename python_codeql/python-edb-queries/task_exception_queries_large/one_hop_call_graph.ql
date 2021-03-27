import python

// 1-hop call graphs from functions that contain an Exception
// and the called function if both in source
// and the exception statement itself
from FunctionInvocation fi, Function f1, Function f2, ExceptStmt e
where f1.inSource() 
  and f2.inSource()
  and e.getType().toString() = "HoleException"
  and
  (
    (exists (FunctionInvocation fi2 | fi2.getCaller().getFunction() = f1.getFunctionObject())
      and f1.getFunctionObject() = fi.getCaller().getFunction()
      and f2.getFunctionObject() = fi.getFunction()
      and f1.contains(e)
    )
    or
    ((not exists (FunctionInvocation fi3 | fi3.getCaller().getFunction() = f1.getFunctionObject()))
      and f1 = f2
      and f1.contains(e)
    )
    or
    (not exists (FunctionInvocation fi4 | fi4.getFunction() = f2.getFunctionObject())
      and f1 = f2
      and f2.contains(e)
    )
  )
select f1.getDefinition(), f2.getDefinition()
// f1: function that contains the HoleException
// f2: function that is called by f1