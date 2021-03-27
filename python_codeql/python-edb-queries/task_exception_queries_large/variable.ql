import python
import restrict_boundaries

from Variable id,Scope scope,string name
where variable(id,scope,name)
  and scope.inSource()
  and isInBounds(scope)
select id,
       scope,
       name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")