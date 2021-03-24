import python
import restrict_boundaries

from CmpopList id,Compare parent
where py_cmpop_lists(id,parent)
  and parent.getScope().inSource()
  and isInBounds(parent.getScope())
select id,parent