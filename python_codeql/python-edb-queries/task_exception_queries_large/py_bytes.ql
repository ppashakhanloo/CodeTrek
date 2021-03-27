import python
import restrict_boundaries

from string id,BytesOrStr parent,int idx
where py_bytes(id,parent,idx)
select id.replaceAll("\n", "\\n").replaceAll("\r", "\\r").replaceAll("\t", "\\t"),
       parent,
       idx