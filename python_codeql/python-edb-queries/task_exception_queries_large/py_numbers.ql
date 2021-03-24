import python
import restrict_boundaries

from string id,Num parent,int idx
where py_numbers(id,parent,idx)
  and parent.getScope().inSource()
  and isInBounds(parent.getScope())
select id.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       parent,
       idx