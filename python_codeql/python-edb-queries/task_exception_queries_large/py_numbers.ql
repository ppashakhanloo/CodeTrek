import python

from string id,Num parent,int idx
where py_numbers(id,parent,idx)
  and parent.getScope().inSource()
select id.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       parent,
       idx