import python
import pruning

from LocalVariable id,Scope scope,string name
where variable(id,scope,name)
  and isSourceLocation(scope.getLocation())
  and not exists (Function f | f.getName() = name and f.getScope() = scope)
select id,
       scope,
       name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")
