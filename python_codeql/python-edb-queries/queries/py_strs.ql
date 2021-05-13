import python
import extra_classes

from string id,StrParent parent,int idx
where py_strs(id,parent,idx)
select id.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       parent,
       idx
