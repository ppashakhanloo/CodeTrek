import python
import pruning

from CmpopList id,Compare parent
where py_cmpop_lists(id,parent)
  and isSourceLocation(parent.getLocation())
select id,parent
