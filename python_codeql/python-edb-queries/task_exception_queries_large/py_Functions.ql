import python
import restrict_boundaries

from Function id
where py_Functions(id,id.getParent())
  and id.inSource()
  and isInBounds(id)
select id,id.getParent()