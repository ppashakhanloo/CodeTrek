import python
import restrict_boundaries

from Comprehension id,ComprehensionList parent,int idx
where py_comprehensions(id,parent,idx)
  and id.getScope().inSource()
  and isInBounds(id.getScope())
select id,parent,idx