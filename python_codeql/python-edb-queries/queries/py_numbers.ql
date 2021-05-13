import python
import pruning

from string id,Num parent,int idx
where py_numbers(id,parent,idx)
  and isSourceLocation(parent.getLocation())
select id.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       parent,
       idx
