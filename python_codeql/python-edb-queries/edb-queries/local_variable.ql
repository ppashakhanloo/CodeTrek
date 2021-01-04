import python
import pruning

from LocalVariable id,Scope scope,string name
where variable(id,scope,name)
select id,
       scope,
       name.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t")
